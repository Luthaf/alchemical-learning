import torch
import numpy as np

from equistore import TensorMap, TensorBlock, Labels


class CombineRadialSpecies(torch.nn.Module):
    def __init__(self, n_species, max_radial, n_combined_basis, *, explicit_combining_matrix=None):
        super().__init__()
        self.n_species = n_species
        self.max_radial = max_radial

        if explicit_combining_matrix is None:
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
        assert spherical_expansion.property_names == ("neighbor_species", "n")

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
