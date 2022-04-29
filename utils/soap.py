import numpy as np
import torch
import math

from equistore import Labels, TensorBlock, TensorMap


class PowerSpectrum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spherical_expansion: TensorMap) -> TensorMap:
        # Make sure that the expansion coefficients have the correct set of keys
        # associated with 1-center expansion coefficients.
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)

        blocks = []
        for (l,), spx_1 in spherical_expansion:
            spx_2 = spherical_expansion.block(spherical_harmonics_l=l)

            # with the same central species, we should have the same samples
            assert np.all(spx_1.samples == spx_2.samples)

            factor = 1.0 / math.sqrt(2 * l + 1)
            properties = Labels(
                names=[f"{name}_1" for name in spx_1.properties.names]
                + [f"{name}_2" for name in spx_2.properties.names],
                values=np.array(
                    [
                        properties_1.tolist() + properties_2.tolist()
                        for properties_1 in spx_1.properties
                        for properties_2 in spx_2.properties
                    ],
                    dtype=np.int32,
                ),
            )

            # Compute the invariants by summation and store the results
            data = factor * torch.einsum("ima, imb -> iab", spx_1.values, spx_2.values)

            block = TensorBlock(
                values=data.reshape(data.shape[0], -1),
                samples=spx_1.samples,
                components=[],
                properties=properties,
            )

            if spx_1.has_gradient("positions"):
                n_properties = block.values.shape[1]
                gradient_1 = spx_1.gradient("positions")
                gradient_2 = spx_2.gradient("positions")

                if len(gradient_1.samples) == 0 or len(gradient_2.samples) == 0:
                    continue

                gradients_samples = np.unique(
                    np.concatenate([gradient_1.samples, gradient_2.samples])
                )
                gradients_samples = gradients_samples.view(np.int32).reshape(-1, 3)

                gradients_samples = Labels(
                    names=gradient_1.samples.names, values=gradients_samples
                )

                gradients_sample_mapping = {
                    tuple(sample): i for i, sample in enumerate(gradients_samples)
                }

                gradient_data = torch.zeros(
                    (gradients_samples.shape[0], 3, n_properties),
                    device=gradient_1.data.device,
                )

                gradient_data_1 = factor * torch.einsum(
                    "ixma, imb -> ixab",
                    gradient_1.data,
                    spx_2.values[gradient_1.samples["sample"].tolist(), :, :],
                ).reshape(gradient_1.samples.shape[0], 3, -1)

                for sample, row in zip(gradient_1.samples, gradient_data_1):
                    new_row = gradients_sample_mapping[tuple(sample)]
                    gradient_data[new_row, :, :] += row

                gradient_data_2 = factor * torch.einsum(
                    "ima, ixmb -> ixab",
                    spx_1.values[gradient_2.samples["sample"].tolist(), :, :],
                    gradient_2.data,
                ).reshape(gradient_2.samples.shape[0], 3, -1)

                for sample, row in zip(gradient_2.samples, gradient_data_2):
                    new_row = gradients_sample_mapping[tuple(sample)]
                    gradient_data[new_row, :, :] += row

                assert gradient_1.components[0].names == ("gradient_direction",)
                block.add_gradient(
                    "positions",
                    gradient_data,
                    gradients_samples,
                    [gradient_1.components[0]],
                )

            blocks.append(block)

        descriptor = TensorMap(spherical_expansion.keys, blocks)
        descriptor.keys_to_properties("spherical_harmonics_l", sort_samples=False)
        return descriptor
