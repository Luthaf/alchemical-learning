import torch
import numpy as np

from equistore import TensorMap, TensorBlock, Labels


class CombineRadialSpecies(torch.nn.Module):
    def __init__(self, n_species, max_radial, n_combined_basis):
        super().__init__()
        self.combining_matrix = torch.nn.Parameter(
            torch.rand((max_radial * n_species, n_combined_basis))
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
