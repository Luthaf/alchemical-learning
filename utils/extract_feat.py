from ase.io import read
import torch
#torch.set_default_dtype(torch.float64)

from utils.linear import LinearModel
from utils.model import CombinedPowerSpectrum
from copy import copy
from utils.multi_species_mlp import EmbeddingFactory
from utils.operations import remove_gradient, StructureMap
from time import time

class SimpleMLP(torch.nn.Module):
    """ A simple MLP 
    """

    # TODO: add n_hidden_layers, activation function option
    def __init__(self, dim_input: int, dim_output: int, layer_size: int, nn_architecture: torch.nn.Sequential = None) -> None:
        super().__init__()


        """nn_architecture overwrites the default 
        """

        self.layer_size = layer_size
        self.dim_input = dim_input
        self.dim_output = dim_output

        if nn_architecture is None:
            self.nn = torch.nn.Sequential(
                torch.nn.Linear(self.dim_input, self.layer_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.layer_size, self.layer_size),
                torch.nn.Tanh(),
            )
        else:
            self.nn = nn_architecture

    def forward(self,x: torch.tensor) -> torch.tensor:
        return self.nn(x)

class MultiMLP(torch.nn.Module):    
    """ A Multi MLP that contains N_species * SimpleMLPs
    """
    def __init__(self, dim_input: int, dim_output: int, layer_size: int, species: int, nn_architecture: torch.nn.Module=None) -> None:
        super().__init__()

        self.dim_output = dim_output
        self.dim_input = dim_input
        self.layer_size = layer_size
        self.species = species 
        self.n_species = len(self.species)

        # initialize as many SimpleMLPs as atomic species
        if nn_architecture is None:
            self.species_nn = torch.nn.ModuleList([ SimpleMLP(dim_input,dim_output,layer_size) for _ in self.species])
        else:
            self.species_nn = torch.nn.ModuleList([ SimpleMLP(dim_input,dim_output,layer_size,nn_architecture(self.dim_input,self.layer_size,self.dim_output)) for _ in self.species])
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return torch.cat([nn(x) for nn in self.species_nn],dim=1)


class MultiMLP_skip(MultiMLP):
    """ A Multi MLP that contains N_species * SimpleMLPs
        This Implementation does only batchwise evaluation of neural networks?
        As this implementation skips 
    """
    
    def forward(self, x: torch.tensor, batch_z: torch.tensor) -> torch.tensor:
        #will this work with autograd? -> I think it does
        
        #get the unique zs in batch
        unique_z_in_batch = torch.unique(batch_z)

        #initializes an empty torch tensor of shape (N_samples,N_species)
        model_out = [] #torch.empty((x.shape[0],x.shape[1],self.n_species))

        #loops over n_total_species
        for n, (z, nn) in enumerate(zip(self.species, self.species_nn)):
            # if a z is in a global batch -> then use the NN_central_species on the X_central species
            # fill the rest with zeros

            
            if z in unique_z_in_batch:
                mask = (batch_z == z).flatten()
                model_out.append(nn(x[mask,:]))


        model_return = torch.ones(x.shape[0],model_out[0].shape[1])

        
        for n, (z, nn) in enumerate(zip(self.species, self.species_nn)):

            if z in unique_z_in_batch:
               mask = (batch_z == z).flatten()
               model_return[mask,:] = model_out[n] #-> comes as a batch of (N_sample,N_hidden)
            #else: if z is not in batch at all, simply fill everything with zeros

        return model_return #, model_out

class MultiSpeciesMLP_skip(torch.nn.Module):
    
    """ Implements a MultiSpecies Behler Parinello neural network
    This implementation should scale O(Natoms) as it skips the neural network evaluations that would be otherwise only multiplied with zeros
    """

    def __init__(self, species, n_in, n_out, n_hidden, nn_architecture=None) -> None:
        
        super().__init__()

        #just a precaution
        species = copy(species)
        species.sort()

        #print(species)
    
        #TODO: Implement this properly in the MultiSpeciesMLP class

        self.n_out = n_out
        self.species = species

        # if we want to skip the NN evaluations the Embedding has to be non trainable
        # therefore -> MultiMLP_skip has no trainable kwargs and one_hot in EmbeddingFactory is always true
        # TODO: Implement this properly in the MultiSpeciesMLP class

        self.nn = MultiMLP_skip(n_in,n_out,n_hidden,species,nn_architecture)
        self.embedding = EmbeddingFactory(species, True)
        self.embedding.requires_grad_ = False

    def forward(self, x: torch.tensor, z:torch.tensor) -> torch.tensor:
        # here the embedding multiplication should only introduce a minor overhead 
        return self.nn(x,z)

class NNModel(torch.nn.Module):
    def __init__(self, layer_size=100):
        super().__init__()

        self.nn = None
        self.layer_size = layer_size

    # build a combined 
    def initialize_model_weights(self, descriptor, energies, forces=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        X = descriptor.block().values

        # initialize nn with zero weights ??
        def init_zero_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        # 

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(X.shape[-1], self.layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.layer_size, self.layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.layer_size, 1),
        )

    def forward(self, descriptor, with_forces=False):
        if self.nn is None:
            raise Exception("call initialize_weights first")

        ps_block = descriptor.block()
        ps_tensor = ps_block.values

        if with_forces:
            # TODO(guillaume): can this have unintended side effects???
            ps_tensor.requires_grad_(True)

        structure_map, new_samples, _ = StructureMap(
            ps_block.samples["structure"], ps_tensor.device
        )

        # (pstensor: shape (Natoms,Ddescriptor) -> (Natoms,))

        # will be (Natoms,Ddescriptor) -> (Natoms,Nensembles) #what if we did deep ensemble?
        # (Natoms,Nensemble,2) ?
        nn_per_atom = self.nn(ps_tensor)
        
        #structure is actually atomic envs
        nn_per_structure = torch.zeros((len(new_samples), 1), device=ps_tensor.device)
        
        #adds atomic contributions per structure


        #TODO: this will need an additional axis (Natom,) -> (Nstruct)
        # will eventually be (Natom, )

        # index add -> uncertainties are added aswell?
        nn_per_structure.index_add_(0, structure_map, nn_per_atom)

        energies = nn_per_structure
        
        if with_forces:
            
            # computes dnn/dg for dnn/dg dg/dx
            nn_grads = torch.autograd.grad(
                nn_per_structure,
                ps_tensor,
                grad_outputs=torch.ones_like(nn_per_structure),
                create_graph=True,
                retain_graph=True,
            )


            ps_gradient = descriptor.block().gradient("positions")
            ps_tensor_grad = ps_gradient.data.reshape(-1, 3, ps_tensor.shape[-1])

            gradient_samples_Aj = np.asarray(
                ps_gradient.samples[["structure", "atom"]], dtype=tuple
            )

            #why is a unique gradient necessary
            unique_gradient, unique_gradient_idx = np.unique(
                gradient_samples_Aj, return_index=True
            )
            # new_gradient_samples = gradient_samples_Aj[np.sort(unique_gradient_idx)]

            # the logic is analogous to that for the structures: we have to map
            # positions in the full (A,i,j) vector to the position where they
            # will have to be accumulated
            gradient_replace_rule = dict(
                zip(unique_gradient, range(len(unique_gradient)))
            )


            gradient_map = torch.tensor(
                [gradient_replace_rule[i] for i in gradient_samples_Aj],
                dtype=torch.long,
                device=ps_tensor.device,
            )

            new_gradient_data = torch.zeros(
                (len(unique_gradient), 3, 1),
                device=ps_tensor.device,
            )
            # ... and then contracting the gradients is just one call
            nn_per_atom_forces = -torch.sum(
                ps_tensor_grad * nn_grads[0][gradient_map][:, None, :], -1
            )

            #why the index add here ?
            new_gradient_data.index_add_(
                0, gradient_map, nn_per_atom_forces[:, :, None]
            )
            forces = new_gradient_data.reshape(-1, 3)
        else:
            forces = None
        return energies, forces

class NNModelSPeciesWise(torch.nn.Module):
    def __init__(self, layer_size=100):
        super().__init__()

        self.nn = None
        self.layer_size = layer_size
        # for now we only want to predict energies
        self.n_out = 1

    # build a combined 
    def initialize_model_weights(self, descriptors, energies, forces=None, seed=None, nn_architecture=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        for descriptor in descriptors:
            print(descriptor.block().values.shape[-1])

        n_feats = sum([ descriptor.block().values.shape[-1] for descriptor in descriptors])
        z = torch.tensor(descriptors[0].block().samples["species_center"])

        species_unique = torch.unique(z).tolist()


        # initialize nn with zero weights ??
        def init_zero_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        n_feat_descriptor = n_feats


        #MultiSpeciesMLP_skip: feat --> species wise NN, skipping evals --> atomic contributions out
        if nn_architecture is None:
            self.nn = MultiSpeciesMLP_skip(species_unique,n_feat_descriptor,self.n_out,self.layer_size)
        else:
            self.nn = MultiSpeciesMLP_skip(species_unique,n_feat_descriptor,self.n_out,self.layer_size,nn_architecture=nn_architecture)

    def forward(self, descriptors, with_forces=False):
        if self.nn is None:
            raise Exception("call initialize_weights first")

        ps_block = descriptors[0].block()
        ps_tensor = torch.cat([ descriptor.block().values for descriptor in descriptors],dim=1)

        # obtaining central species of batch
        ps_z = torch.tensor(ps_block.samples["species_center"])

        if with_forces:
            # TODO(guillaume): can this have unintended side effects???
            ps_tensor.requires_grad_(True)

        structure_map, new_samples, _ = StructureMap(
            ps_block.samples["structure"], ps_tensor.device
        )

        nn_per_atom = self.nn(ps_tensor, ps_z)
        
        #structure is actually atomic envs
        print((len(new_samples), nn_per_atom.shape[1]))
        nn_per_structure = torch.zeros((len(new_samples), nn_per_atom.shape[1]), device=ps_tensor.device)
        
        #adds atomic contributions per structure
        #nn_per_structure.index_add_(1, structure_map, nn_per_atom)

        feats = nn_per_atom 
        #structure

        return feats








class SoapBpnn(torch.nn.Module):
    def __init__(
        self,
        # combiner for the spherical expansion (mainly tested with alchemical
        # combination)
        combiner,
        # Regularizer for the 1-body/composition model. Set to None to remove
        # the 1-body model from the fit
        composition_regularizer,
        # Number of layers for the neural network applied on top of power spectrum
        # Set to None to use a pure linear model
        nn_layer_size=None,
        # list of atomic types to explode the power spectrum. None to use same
        # model for all centers (default)
        ps_center_types=None,
        optimizable_weights=True,
        random_initial_weights=True,
        nn_model_architecture=None,
    ):
        super().__init__()

        if composition_regularizer is None:
            self.composition_model = None
        else:
            self.composition_model = LinearModel(
                regularizer=composition_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        self.power_spectrum = CombinedPowerSpectrum(combiner)
        self.nn_model = NNModelSPeciesWise(nn_layer_size)

        self._neval = 0        
        self._timings = dict(fw_comp = 0.0, fw_pair = 0.0, fw_ps = 0.0, fw_nn = 0.0)

    
    def forward(
        self,
        composition,
        radial_spectrum,
        spherical_expansion,
        forward_forces=False,
    ):
        if not forward_forces:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)

        energies = torch.zeros(
            (len(composition.block().samples), 1),
            device=composition.block().values.device,
        )
        forces = None

        #Model that learns and removes the energy offset, as a function of the structure composition
        if self.composition_model is not None:
            self._timings["fw_comp"] -= time()
            energies_cmp, _ = self.composition_model(composition)
            energies += energies_cmp
            self._timings["fw_comp"] += time()            
            self._timings["fw_pair"] += time()  

        #TODO: Remove this and concatenate RS with SOAP
                    
                              
        
        #TODO: remove the linear part and only apply NN
        self._timings["fw_ps"] -= time()

        #TODO: remove this because we get the SOAPs sthrough rascaline 
        power_spectrum = spherical_expansion #self.power_spectrum(spherical_expansion)

        feats = self.nn_model(
            [power_spectrum, radial_spectrum], with_forces=forward_forces
        )
                   
        return feats

    def initialize_model_weights(
        self,
        composition,
        radial_spectrum,
        spherical_expansion,
        energies,
        forces=None,
        seed=None,
        nn_architecture=None
    ):
        if forces is None:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)

        if self.composition_model is not None:
            self.composition_model.initialize_model_weights(
                composition, energies, forces, seed
            )

            energies -= self.composition_model(composition)[0]

        if self.nn_model is not None:
            power_spectrum = spherical_expansion #self.power_spectrum(spherical_expansion)
            self.nn_model.initialize_model_weights([power_spectrum, radial_spectrum], energies, forces, seed, nn_architecture)
