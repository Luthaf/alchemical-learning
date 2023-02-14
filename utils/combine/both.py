import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap


class CombineRadialSpecies(torch.nn.Module):
    def __init__(
        self,
        n_species,
        max_radial,
        n_combined_basis,
        seed=None,
        *,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.n_species = n_species
        self.max_radial = max_radial

        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((max_radial * n_species, n_combined_basis))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineRadialSpecies(
            self.n_species,
            self.max_radial,
            self.combining_matrix.shape[1],
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        n_properties, n_combined_basis = self.combining_matrix.shape

        properties = Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(n_combined_basis)],
                dtype=np.int32,
            ),
        )

        blocks = []
        for _, block in spherical_expansion:
            new_block = TensorBlock(
                values=block.values @ self.combining_matrix,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ self.combining_matrix,
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)

class CombineRadialSpeciesWithCentralSpecies(torch.nn.Module):
    def __init__(
        self,
        all_species,
        max_radial,
        n_combined_basis,
        n_pseudo_central_species=None,
        seed=None,
        *,
        explicit_linear_params=None,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.all_species = all_species
        self.n_species = len(all_species)
        self.max_radial = max_radial
        self.central_species_index = {s: i for i, s in enumerate(all_species)}

        if n_pseudo_central_species is None:
            self.n_pseudo_central_species = self.n_species
        else:
            self.n_pseudo_central_species = min(n_pseudo_central_species, self.n_species)
        
        if self.n_pseudo_central_species == self.n_species:
            self.linear_params = None
        elif explicit_linear_params is None:
            # It is the coupling matrix
            self.linear_params = torch.nn.Parameter(torch.rand((self.n_species, self.n_pseudo_central_species)))
        else:
            self.register_buffer("linear_params", explicit_linear_params)
        
        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((self.n_pseudo_central_species, max_radial * self.n_species, n_combined_basis))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineRadialSpeciesWithCentralSpecies(
            self.all_species,
            self.max_radial,
            self.combining_matrix.shape[2],
            self.n_pseudo_central_species,
            explicit_linear_params=None
                if self.linear_params is None
                else self.linear_params.clone().detach(),
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l", "species_center")
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        _, _, n_combined_basis = self.combining_matrix.shape

        properties = Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(n_combined_basis)],
                dtype=np.int32,
            ),
        )

        if self.linear_params is None:
            cs_combining_matrix = self.combining_matrix
        else:
            cs_combining_matrix = self.combining_matrix.swapaxes(0, 1)
            cs_combining_matrix = (self.linear_params @ cs_combining_matrix).swapaxes(0, 1)
            # or
            # cs_combining_matrix = torch.einsum('sp,pnq->snq', self.linear_params, self.combining_matrix)

        blocks = []
        for key, block in spherical_expansion:
            cs_i = self.central_species_index[key[1]]
            new_block = TensorBlock(
                values=block.values @ cs_combining_matrix[cs_i],
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ cs_combining_matrix[cs_i],
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        new_tensor_map = TensorMap(spherical_expansion.keys, blocks)
        new_tensor_map.keys_to_samples("species_center")
        return new_tensor_map
