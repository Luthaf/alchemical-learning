import math

# only profile when required
try:
    profile
except NameError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap


class CompositionFeatures(torch.nn.Module):
    def __init__(self, all_species, device="cpu"):
        super().__init__()
        self.species_dict = {s: i for i, s in enumerate(all_species)}
        self.device = device

    @profile
    def forward(self, frames, frames_i=None):
        data = torch.zeros(
            size=(len(frames), len(self.species_dict)), device=self.device
        )
        for i, f in enumerate(frames):
            for s in f.numbers:
                data[i, self.species_dict[s]] += 1
        properties = Labels(
            names=["n_species"],
            values=np.array(list(self.species_dict.keys()), dtype=np.int32).reshape(
                -1, 1
            ),
        )
        if frames_i is None:
            frames_i = np.arange(len(frames), dtype=np.int32).reshape(-1, 1)
        samples = Labels(names=["structure"], values=frames_i)

        block = TensorBlock(
            values=data, samples=samples, components=[], properties=properties
        )
        descriptor = TensorMap(Labels.single(), blocks=[block])
        return descriptor


class PowerSpectrum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @profile
    def forward(self, spherical_expansion: TensorMap) -> TensorMap:
        # Make sure that the expansion coefficients have the correct set of keys
        # associated with 1-center expansion coefficients.
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)

        soap_start = 0
        lmax = 0
        for (l,), spx_1 in spherical_expansion:
            soap_start += len(spx_1.properties) ** 2
            lmax = max(l, lmax)
        data = torch.empty((len(spx_1.samples), soap_start), device=spx_1.values.device)

        if spx_1.has_gradient("positions"):
            data_grad = torch.empty(
                (len(spx_1.gradient("positions").samples), 3, soap_start),
                device=spx_1.values.device,
            )
        soap_start = 0  # resets counter
        properties_names = (
            [f"{name}_1" for name in spx_1.properties.names]
            + [f"{name}_2" for name in spx_1.properties.names]
            + ["spherical_harmonics_l"]
        )
        properties_values = []
        for l in range(lmax+1): # loops over l to ensure consistent order, independent on key storage
            spx_1 = spherical_expansion.block(spherical_harmonics_l=l)
            spx_2 = spherical_expansion.block(spherical_harmonics_l=l)

            # with the same central species, we should have the same samples
            assert np.all(spx_1.samples == spx_2.samples)

            factor = 1.0 / math.sqrt(2 * l + 1)
            properties_values.append(
                [
                    properties_1.tolist() + properties_2.tolist() + (l,)
                    for properties_1 in spx_1.properties
                    for properties_2 in spx_2.properties
                ]
            )
            soap_stop = soap_start + len(properties_values[-1])

            # Compute the invariants by summation and store the results
            data[:, soap_start:soap_stop] = torch.einsum(
                "ima, imb -> iab", factor * spx_1.values, spx_2.values
            ).reshape(data.shape[0], -1)

            if spx_1.has_gradient("positions"):
                gradient_1 = spx_1.gradient("positions")
                gradient_2 = spx_2.gradient("positions")

                if len(gradient_1.samples) == 0 or len(gradient_2.samples) == 0:
                    continue

                # we know all samples are the same for this project, use this knowledge
                assert np.all(gradient_1.samples == gradient_2.samples)
                gradients_samples = gradient_1.samples

                data_grad[:, :, soap_start:soap_stop] = torch.einsum(
                    "ixma, imb -> ixab",
                    gradient_1.data,
                    factor * spx_2.values[gradient_1.samples["sample"].tolist(), :, :],
                ).reshape(gradients_samples.shape[0], 3, -1)

                data_grad[:, :, soap_start:soap_stop] += torch.einsum(
                    "ima, ixmb -> ixab",
                    factor * spx_1.values[gradient_2.samples["sample"].tolist(), :, :],
                    gradient_2.data,
                ).reshape(gradients_samples.shape[0], 3, -1)

                assert gradient_1.components[0].names == ("direction",)

            soap_start = soap_stop

        block = TensorBlock(
            values=data,
            samples=spx_1.samples,
            components=[],
            properties=Labels(
                names=properties_names,
                values=np.asarray(np.vstack(properties_values), dtype=np.int32),
            ),
        )
        if spx_1.has_gradient("positions"):
            block.add_gradient(
                "positions",
                data_grad,
                gradients_samples,
                [gradient_1.components[0]],
            )
        descriptor = TensorMap(Labels.single(), [block])
        return descriptor
