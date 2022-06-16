import math

import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap

class CompositionFeatures(torch.nn.Module):
    def __init__(self, 
        all_species
    ):
        super().__init__()
        self.species_dict = {s:i for i,s in enumerate(all_species)}    
    
    def forward(self, frames) -> TensorMap:
        
        data = torch.zeros(size=(len(frames), len(self.species_dict)))
        for i, f in enumerate(frames):
            for s in f.numbers:
                data[i,self.species_dict[s]] += 1
        properties = Labels(
            names=["n_species"],
            values = np.array(list(self.species_dict.keys()), dtype=np.int32).reshape(-1,1)
        )
        
        samples = Labels( names = ["structure"], values = np.arange(len(frames),dtype=np.int32).reshape(-1,1)) 
        
        block = TensorBlock(
            values=data, 
            samples=samples,
            components=[],
            properties=properties)
        descriptor = TensorMap(Labels.single(),blocks=[block])
        return descriptor



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
                gradient_1 = spx_1.gradient("positions")
                gradient_2 = spx_2.gradient("positions")

                if len(gradient_1.samples) == 0 or len(gradient_2.samples) == 0:
                    continue

                # we know all samples are the same for this project, use this knowledge
                assert np.all(gradient_1.samples == gradient_2.samples)
                gradients_samples = gradient_1.samples

                gradient_data = factor * torch.einsum(
                    "ixma, imb -> ixab",
                    gradient_1.data,
                    spx_2.values[gradient_1.samples["sample"].tolist(), :, :],
                ).reshape(gradients_samples.shape[0], 3, -1)

                gradient_data += factor * torch.einsum(
                    "ima, ixmb -> ixab",
                    spx_1.values[gradient_2.samples["sample"].tolist(), :, :],
                    gradient_2.data,
                ).reshape(gradients_samples.shape[0], 3, -1)

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
