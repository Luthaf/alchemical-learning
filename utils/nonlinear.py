import numpy as np
import torch

from utils.operations import StructureMap
from utils.multi_species_mlp import MultiSpeciesMLP_skip

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

        nn_per_atom = self.nn(ps_tensor)
        
        #structure is actually atomic envs
        nn_per_structure = torch.zeros((len(new_samples), 1), device=ps_tensor.device)
        
        #adds atomic contributions per structure
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
    def initialize_model_weights(self, descriptors, energies, forces=None, seed=None):
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

        # 
        n_feat_descriptor = n_feats


        #MultiSpeciesMLP_skip: feat --> species wise NN, skipping evals --> atomic contributions out
        self.nn = MultiSpeciesMLP_skip(species_unique,n_feat_descriptor,self.n_out,self.layer_size)

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
        nn_per_structure = torch.zeros((len(new_samples), 1), device=ps_tensor.device)
        
        #adds atomic contributions per structure
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

            ps_gradient = descriptors[0].block().gradient("positions")
            ps_tensor_grad = torch.cat([ 
            descriptor.block().gradient("positions").data.reshape(-1, 3, descriptor.block().values.shape[-1])
            for descriptor in descriptors ],dim=2)

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
