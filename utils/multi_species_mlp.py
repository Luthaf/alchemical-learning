from typing import List
import numpy as np
import torch
from copy import copy
from dataclasses import dataclass

@dataclass
class EquistoreDummy:
    z: torch.Tensor
    val: torch.Tensor
    idx: torch.Tensor

def EmbeddingFactory(elements:List[int],one_hot:bool) -> torch.nn.Embedding:
    """Returns an Embedding of dim max_Z,n_unique_elements
    max_Z = 9, n_unique = 2, elements = [1,8]
    Embedding(tensor([8])) -> tensor([0.0,1.0]) (if one hot)
    """
    
    # embedding "technically" starts at zero, Z at one
    max_int = max(elements) + 1
    n_species = len(elements)

    #randomly initialize the Embedding
    #TODO: add a initialize_weights routine
    #TODO: maybe solve it with a decorator?
    embedding = torch.nn.Embedding(max_int,n_species)
    
    # If the embedding is one-hot, the weight matrix is diagonal
    if one_hot:
        weights = torch.zeros(max_int, n_species)
        
        for idx, Z in enumerate(elements):
            weights[Z, idx] = 1.0

        embedding.weight.data = weights

    return embedding

class SimpleMLP(torch.nn.Module):
    """ A simple MLP 
    """

    # TODO: add n_hidden_layers, activation function option
    def __init__(self, dim_input: int, dim_output: int, layer_size: int) -> None:
        super().__init__()

        self.layer_size = layer_size
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.dim_input, self.layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.layer_size, self.layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.layer_size, self.dim_output),
        )
    def forward(self,x: torch.tensor) -> torch.tensor:
        return self.nn(x)

class MultiMLP(torch.nn.Module):    
    """ A Multi MLP that contains N_species * SimpleMLPs
    """
    def __init__(self, dim_input: int, dim_output: int, layer_size: int, species: int) -> None:
        super().__init__()

        self.dim_output = dim_output
        self.dim_input = dim_input
        self.layer_size = layer_size
        self.species = species 
        self.n_species = len(self.species)

        # initialize as many SimpleMLPs as atomic species
        self.species_nn = torch.nn.ModuleList([ SimpleMLP(dim_input,dim_output,layer_size) for _ in self.species])
    
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
        model_out = torch.empty((x.shape[0],self.n_species))

        #loops over n_total_species
        for n, (z, nn) in enumerate(zip(self.species, self.species_nn)):
            
            # if a z is in a global batch -> then use the NN_central_species on the X_central species
            # fill the rest with zeros
            if z in unique_z_in_batch:
                model_out[batch_z == z, n] = nn(x[batch_z == z]).flatten()
                model_out[batch_z != z, n] = torch.zeros(x[batch_z != z].shape[0])
            
            #else: if z is not in batch at all, simply fill everything with zeros
            else:
                model_out[:, n] = torch.zeros(x.shape[0])

        return model_out


class MultiSpeciesMLP(torch.nn.Module):
    
    """ Implements a MultiSpecies Behler Parinello neural network
    This implementation scales O(Nspecies*Natoms), but it has a learnable weight matrix, that combines species wise energies
    """

    def __init__(self, species, n_in, n_out, n_hidden, one_hot, embedding_trainable) -> None:
        
        super().__init__()
        
        #just a precaution
        species = copy(species)
        species.sort()

        #print(species)

        self.species = species
        self.nn = MultiMLP(n_in,n_out,n_hidden,species)
        self.embedding = EmbeddingFactory(species, one_hot)

        if not embedding_trainable:
            self.embedding.requires_grad_ = False
            

    def forward(self, descriptor: EquistoreDummy) -> torch.tensor:
        
        x = descriptor.val 
        z = descriptor.z # something like descriptor.

        #The embedding serves a a multiplicative "mask" -> not so nice overall complexity scales as O(N_species*N_samples)
        # whereas an implementation that could "skip" NN evaluations should only scale as O(N_samples)
        return torch.sum(self.nn(x) * self.embedding(z),dim=1)


class MultiSpeciesMLP_skip(torch.nn.Module):
    
    """ Implements a MultiSpecies Behler Parinello neural network
    This implementation should scale O(Natoms) as it skips the neural network evaluations that would be otherwise only multiplied with zeros
    """

    def __init__(self, species, n_in, n_out, n_hidden) -> None:
        
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

        self.nn = MultiMLP_skip(n_in,n_out,n_hidden,species)
        self.embedding = EmbeddingFactory(species, True)
        self.embedding.requires_grad_ = False

    def forward(self, x: torch.tensor, z:torch.tensor) -> torch.tensor:
        # here the embedding multiplication should only introduce a minor overhead 
        return torch.sum(self.nn(x,z) * self.embedding(z),dim=1,keepdim=True)   


    """
    def forward(self, descriptor: EquistoreDummy) -> torch.tensor:
        
        x = descriptor.val  # something like descriptor.
        z = descriptor.z 

        # here the embedding multiplication should only introduce a minor overhead 
        return torch.sum(self.nn(x,z) * self.embedding(z),dim=1)
    """

class MultiSpeciesMLP_skip_w_index_add(MultiSpeciesMLP_skip):
    
    """ For testing purposes I have added the atomic-contributions to structure wise properties addition
    """
    
    def forward(self, descriptor: EquistoreDummy) -> torch.tensor:
                
        x = descriptor.val  
        z = descriptor.z 
        idx = descriptor.idx

        x.requires_grad_(True)
        num_structures = len(torch.unique(idx))
        
        structure_wise_properties = torch.zeros((num_structures,self.n_out))

        # In the summation of the atomic contirbutions the dimensions should be kept for autograds
        atomic_contributions = torch.sum(self.nn(x,z) * self.embedding(z),dim=1,keepdim=True)

        print(atomic_contributions.flatten())
        
        structure_wise_properties.index_add_(0,idx,atomic_contributions)

        nn_grads = torch.autograd.grad(
                structure_wise_properties,
                x,
                grad_outputs=torch.ones_like(structure_wise_properties),
                create_graph=True,
                retain_graph=True,
            )

        return structure_wise_properties, nn_grads

