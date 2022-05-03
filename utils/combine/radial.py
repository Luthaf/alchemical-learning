import torch
import numpy as np

from equistore import TensorMap, Labels, TensorBlock


class CombineRadial(torch.nn.Module):
    def __init__(self, max_radial, n_combined_radial):
        super().__init__()
        self.combining_matrix = torch.nn.Parameter(
            torch.rand((max_radial, n_combined_radial))
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("neighbor_species", "n")

        n_radial, n_combined_radial = self.combining_matrix.shape

        radial = np.unique(spherical_expansion.block(0).properties["n"])
        assert len(radial) == n_radial

        species = np.unique(spherical_expansion.block(0).properties["neighbor_species"])
        n_species = len(species)

        properties = Labels(
            names=["neighbor_species", "n"],
            values=np.array(
                [[s, -n] for n in range(n_combined_radial) for s in species],
                dtype=np.int32,
            ),
        )

        blocks = []
        for _, block in spherical_expansion:
            n_samples, n_components, _ = block.values.shape
            data = block.values.reshape(n_samples, n_components, n_species, n_radial)
            data = data @ self.combining_matrix
            data = data.reshape(n_samples, n_components, n_species * n_combined_radial)

            new_block = TensorBlock(
                values=data,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")

                n_grad_samples, n_cartesian, n_spherical, _ = gradient.data.shape

                data = gradient.data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_species,
                    n_radial,
                )

                data = data @ self.combining_matrix

                data = data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_species * n_combined_radial,
                )

                new_block.add_gradient(
                    "positions",
                    data,
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)
