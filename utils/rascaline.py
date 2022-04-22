import copy
import numpy as np

from rascaline import SphericalExpansion

from equistore import Labels, TensorBlock, TensorMap


class RascalineSphericalExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames) -> TensorMap:
        max_radial = self._hypers["max_radial"]
        max_angular = self._hypers["max_angular"]

        calculator = SphericalExpansion(**self._hypers)
        descriptor = calculator.compute(frames)

        old_samples = descriptor.samples
        old_gradient_samples = descriptor.gradients_samples
        values = descriptor.values.reshape(descriptor.values.shape[0], -1, max_radial)

        species = np.unique(old_samples[["species_center", "species_neighbor"]])
        keys = Labels(
            names=["spherical_harmonics_l", "center_species", "neighbor_species"],
            values=np.array(
                [
                    [l, species_center, species_neighbor]
                    for l in range(max_angular + 1)
                    for species_center, species_neighbor in species
                ],
                dtype=np.int32,
            ),
        )

        properties = Labels(
            names=["n"],
            values=np.array([[n] for n in range(max_radial)], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(max_angular + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        if descriptor.gradients is not None:
            has_gradients = True
            gradients = descriptor.gradients.reshape(
                descriptor.gradients.shape[0], -1, max_radial
            )
        else:
            has_gradients = False

        blocks = []
        for l, center_species, neighbor_species in keys:
            centers = np.unique(
                old_samples[old_samples["species_center"] == center_species][
                    ["structure", "center"]
                ]
            )
            center_map = {tuple(center): i for i, center in enumerate(centers)}

            block_data = np.zeros((len(centers), 2 * l + 1, max_radial))

            mask = np.logical_and(
                old_samples["species_center"] == center_species,
                old_samples["species_neighbor"] == neighbor_species,
            )
            for sample_i in np.where(mask)[0]:
                new_sample_i = center_map[
                    tuple(old_samples[sample_i][["structure", "center"]])
                ]
                block_data[new_sample_i, :, :] = values[sample_i, lm_slices[l], :]

            samples = Labels(
                names=["structure", "center"],
                values=np.array(
                    [[structure, center] for structure, center in center_map.keys()],
                    dtype=np.int32,
                ),
            )
            spherical_component = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            block_gradients = None
            gradient_samples = None
            if has_gradients:
                gradient_samples = []
                block_gradients = []
                for sample_i in np.where(mask)[0]:
                    gradient_mask = np.logical_and(
                        old_gradient_samples["sample"] == sample_i,
                        old_gradient_samples["spatial"] == 0,
                    )

                    new_sample_i = center_map[
                        tuple(old_samples[sample_i][["structure", "center"]])
                    ]

                    for grad_index in np.where(gradient_mask)[0]:
                        block_gradients.append(
                            gradients[
                                None, grad_index : grad_index + 3, lm_slices[l], :
                            ]
                        )

                        structure = old_samples[sample_i]["structure"]
                        atom = old_gradient_samples[grad_index][["atom"]][0]

                        gradient_samples.append((new_sample_i, structure, atom))

                if len(gradient_samples) != 0:
                    block_gradients = np.vstack(block_gradients)
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom"],
                        values=np.vstack(gradient_samples).astype(np.int32),
                    )
                else:
                    block_gradients = np.zeros(
                        (0, 3, spherical_component.shape[0], properties.shape[0])
                    )
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom"],
                        values=np.zeros((0, 3), dtype=np.int32),
                    )

            block = TensorBlock(
                values=block_data,
                samples=samples,
                components=[spherical_component],
                properties=properties,
            )

            if block_gradients is not None:
                spatial_component = Labels(
                    names=["gradient_direction"],
                    values=np.array([[0], [1], [2]], dtype=np.int32),
                )
                gradient_components = [spatial_component, spherical_component]
                block.add_gradient(
                    "positions",
                    block_gradients,
                    gradient_samples,
                    gradient_components,
                )

            blocks.append(block)

        return TensorMap(keys, blocks)
