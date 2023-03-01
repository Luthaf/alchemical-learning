from typing import Optional

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


def normalize(descriptor):
    blocks = []
    for _, block in descriptor:
        # only deal with invariants for now
        assert len(block.components) == 0
        assert len(block.values.shape) == 2

        norm = torch.linalg.norm(block.values, dim=1)
        normalized_values = block.values / norm[:, None]

        new_block = TensorBlock(
            values=normalized_values,
            samples=block.samples,
            components=[],
            properties=block.properties,
        )

        if block.has_gradient("positions"):
            gradient = block.gradient("positions")

            samples = gradient.samples["sample"].tolist()
            gradient_data = gradient.data / norm[samples, None, None]

            # gradient of x_i = X_i / N_i is given by
            # 1 / N_i \grad X_i - x_i [x_i @ 1 / N_i \grad X_i]
            # for sample_i, (sample, _, _) in enumerate(gradient.samples):
            dot = torch.einsum(
                "ixa, ia, ib -> ixb",
                gradient_data,
                normalized_values[samples],
                normalized_values[samples],
            )
            gradient_data = gradient_data - dot

            new_block.add_gradient(
                "positions", gradient_data, gradient.samples, gradient.components
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


def StructureMap(samples_structure, device="cpu"):
    unique_structures, unique_structures_idx = np.unique(
        samples_structure, return_index=True
    )
    new_samples = samples_structure[np.sort(unique_structures_idx)]
    # we need a list keeping track of where each atomic contribution goes
    # (e.g. if structure ids are [3,3,3,1,1,1,6,6,6] that will be stored as
    # the unique structures [3, 1, 6], structure_map will be
    # [0,0,0,1,1,1,2,2,2]
    replace_rule = dict(zip(unique_structures, range(len(unique_structures))))
    structure_map = torch.tensor(
        [replace_rule[i] for i in samples_structure],
        dtype=torch.long,
        device=device,
    )
    return structure_map, new_samples, replace_rule


class SumStructuresAutograd(torch.autograd.Function):
    @staticmethod
    @profile
    def forward(
        ctx,
        values: torch.Tensor,
        samples: Labels,
        gradient_data: Optional[torch.Tensor],
        gradient_samples: Optional[Labels],
    ):

        # get the unique entries in samples["structure"] without
        # sorting the result (that would break pytorch dataloader shuffling)
        samples_structure = samples["structure"]
        # unique_structures, unique_structures_idx = np.unique(
        #    samples_structure, return_index=True
        # )
        # new_samples = samples_structure[np.sort(unique_structures_idx)]
        # we need a list keeping track of where each atomic contribution goes
        # (e.g. if structure ids are [3,3,3,1,1,1,6,6,6] that will be stored as
        # the unique structures [3, 1, 6], structure_map will be
        # [0,0,0,1,1,1,2,2,2]
        # replace_rule = dict(zip(unique_structures, range(len(unique_structures))))
        # structure_map = torch.tensor(
        #    [replace_rule[i] for i in samples_structure],
        #    dtype=torch.long,
        #    device=values.device,
        # )

        structure_map, new_samples, replace_rule = StructureMap(
            samples_structure, values.device
        )

        new_values = torch.zeros(
            (len(new_samples), *values.shape[1:]),
            device=values.device,
        )
        new_values.index_add_(0, structure_map, values)

        if gradient_data is not None:
            assert gradient_samples is not None

            # here we need to get unique _gradient_ elements, so [A,j] pairs.
            # The i-atom index is summed over and we don't need to know it
            # explicitly. We convert Labels slices to tuples so we can hash them
            # to make a dict
            gradient_samples_Aj = np.asarray(
                gradient_samples[["structure", "atom"]], dtype=tuple
            )
            unique_gradient, unique_gradient_idx = np.unique(
                gradient_samples_Aj, return_index=True
            )
            new_gradient_samples = gradient_samples_Aj[np.sort(unique_gradient_idx)]
            # the logic is analogous to that for the structures: we have to map
            # positions in the full (A,i,j) vector to the position where they
            # will have to be accumulated
            gradient_replace_rule = dict(
                zip(unique_gradient, range(len(unique_gradient)))
            )
            gradient_map = torch.tensor(
                [gradient_replace_rule[i] for i in gradient_samples_Aj],
                dtype=torch.long,
                device=gradient_data.device,
            )

            new_gradient_data = torch.zeros(
                (len(unique_gradient), *gradient_data.shape[1:]),
                device=gradient_data.device,
            )
            # ... and then contracting the gradients is just one call
            new_gradient_data.index_add_(0, gradient_map, gradient_data)

            # builds gradient labels
            ug_array = np.vstack(unique_gradient)
            new_gradient_samples = Labels(
                names=["sample", "structure", "atom"],
                values=np.asarray(
                    np.hstack(
                        [
                            np.asarray(
                                [replace_rule[i] for i in ug_array[:, 0]]
                            ).reshape(-1, 1),
                            ug_array,
                        ]
                    ),
                    dtype=np.int32,
                ),
            )
            ctx.gradient_map = gradient_map
        else:
            new_gradient_data = None
            new_gradient_samples = None

        ctx.structure_map = structure_map
        ctx.save_for_backward(values, gradient_data)

        return new_values, new_samples, new_gradient_data, new_gradient_samples

    @staticmethod
    @profile
    def backward(
        ctx,
        grad_new_values,
        grad_new_samples,
        grad_new_gradient_data,
        grad_new_gradient_samples,
    ):
        grad_values = None
        grad_gradient_data = None
        values, gradient_data = ctx.saved_tensors

        if values.requires_grad:
            grad_values = grad_new_values[ctx.structure_map, ...]

        if gradient_data is not None and gradient_data.requires_grad:
            grad_gradient_data = grad_new_gradient_data[ctx.gradient_map]

        return grad_values, None, grad_gradient_data, None


class SumStructuresCXAutograd(torch.autograd.Function):
    @staticmethod
    @profile
    def forward(
        ctx,
        values: torch.Tensor,
        samples: Labels,
        gradient_data: Optional[torch.Tensor],
        gradient_samples: Optional[Labels],
        unique_types,
    ):

        replace_types = dict(zip(unique_types, range(len(unique_types))))
        # get the unique entries in samples["structure"] without
        # sorting the result (that would break pytorch dataloader shuffling)
        samples_structure = samples["structure"]
        unique_structures, unique_structures_idx = np.unique(
            samples_structure, return_index=True
        )
        new_samples = samples_structure[np.sort(unique_structures_idx)]
        # we need a list keeping track of where each atomic contribution goes
        # (e.g. if structure ids are [3,3,3,1,1,1,6,6,6] that will be stored as
        # the unique structures [3, 1, 6], structure_map will be
        # [0,0,0,1,1,1,2,2,2]
        replace_rule = dict(zip(unique_structures, range(len(unique_structures))))
        structure_map = torch.tensor(
            [replace_rule[i] for i in samples_structure],
            dtype=torch.long,
            device=values.device,
        )
        new_values = torch.zeros(
            (len(new_samples), values.shape[1] * len(unique_types)),
            device=values.device,
        )

        ctx.species_slice = [replace_types[s] for s in samples["species_center"]]
        for t in unique_types:
            idx_t = np.where(samples["species_center"] == t)[0]

            type_start = values.shape[1] * replace_types[t]
            type_stop = values.shape[1] * (1 + replace_types[t])
            new_values[:, type_start:type_stop].index_add_(
                0, structure_map[idx_t], values[idx_t]
            )

        if gradient_data is not None:
            assert gradient_samples is not None

            # here we need to get unique _gradient_ elements, so [A,j] pairs.
            # The i-atom index is summed over and we don't need to know it
            # explicitly. We convert Labels slices to tuples so we can hash them
            # to make a dict
            gradient_samples_Aj = np.asarray(
                gradient_samples[["structure", "atom"]], dtype=tuple
            )
            unique_gradient, unique_gradient_idx = np.unique(
                gradient_samples_Aj, return_index=True
            )
            new_gradient_samples = gradient_samples_Aj[np.sort(unique_gradient_idx)]
            # the logic is analogous to that for the structures: we have to map
            # positions in the full (A,i,j) vector to the position where they
            # will have to be accumulated
            gradient_replace_rule = dict(
                zip(unique_gradient, range(len(unique_gradient)))
            )
            gradient_map = torch.tensor(
                [gradient_replace_rule[i] for i in gradient_samples_Aj],
                dtype=torch.long,
                device=gradient_data.device,
            )

            new_gradient_data = torch.zeros(
                (
                    len(unique_gradient),
                    gradient_data.shape[1],
                    gradient_data.shape[2] * len(unique_types),
                ),
                device=gradient_data.device,
            )

            species_center = samples["species_center"][gradient_samples["sample"]]
            ctx.gradient_slice = [replace_types[s] for s in species_center]
            # ... and then contracting the gradients is just one call
            for t in unique_types:
                idx_t = np.where(species_center == t)[0]

                type_start = gradient_data.shape[-1] * replace_types[t]
                type_stop = gradient_data.shape[-1] * (replace_types[t] + 1)
                new_gradient_data[..., type_start:type_stop].index_add_(
                    0, gradient_map[idx_t], gradient_data[idx_t]
                )

            # builds gradient labels
            ug_array = np.vstack(unique_gradient)
            new_gradient_samples = Labels(
                names=["sample", "structure", "atom"],
                values=np.asarray(
                    np.hstack(
                        [
                            np.asarray(
                                [replace_rule[i] for i in ug_array[:, 0]]
                            ).reshape(-1, 1),
                            ug_array,
                        ]
                    ),
                    dtype=np.int32,
                ),
            )
            ctx.gradient_map = gradient_map
        else:
            new_gradient_data = None
            new_gradient_samples = None

        ctx.structure_map = structure_map
        ctx.save_for_backward(values, gradient_data)

        return new_values, new_samples, new_gradient_data, new_gradient_samples, None

    @staticmethod
    @profile
    def backward(
        ctx,
        grad_new_values,
        grad_new_samples,
        grad_new_gradient_data,
        grad_new_gradient_samples,
        unique_types,
    ):
        grad_values = None
        grad_gradient_data = None
        values, gradient_data = ctx.saved_tensors

        if values.requires_grad:
            grad_values = torch.vstack(
                [
                    grad_new_values[s, values.shape[1] * t : values.shape[1] * (t + 1)]
                    for s, t in zip(ctx.structure_map, ctx.species_slice)
                ]
            ).to(values.device)

        if gradient_data is not None and gradient_data.requires_grad:
            grad_gradient_data = torch.stack(
                [
                    grad_new_gradient_data[
                        s,
                        ...,
                        gradient_data.shape[-1] * t : gradient_data.shape[-1] * (t + 1),
                    ]
                    for s, t in zip(ctx.gradient_map, ctx.gradient_slice)
                ],
                axis=0,
            ).to(values.device)

        return grad_values, None, grad_gradient_data, None, None


class SumStructures(torch.nn.Module):
    def __init__(self, sum_properties=False, explode_centers=None):
        super().__init__()
        self.sum_properties = sum_properties
        self.explode_centers = explode_centers
        if self.sum_properties:
            raise NotImplementedError()

    @profile
    def forward(self, descriptor):
        blocks = []
        for _, block in descriptor:
            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                gradient_data = gradient.data
                gradient_samples = gradient.samples
                gradient_components = gradient.components
            else:
                gradient_data = None
                gradient_samples = None
                gradient_components = None

            properties = block.properties
            if self.explode_centers is None:
                output = SumStructuresAutograd.apply(
                    block.values, block.samples, gradient_data, gradient_samples
                )
                values, samples, gradient_data, gradient_samples = output
            else:
                output = SumStructuresCXAutograd.apply(
                    block.values,
                    block.samples,
                    gradient_data,
                    gradient_samples,
                    self.explode_centers,
                )
                pval = np.asarray(
                    np.hstack(
                        [
                            np.vstack(
                                [
                                    properties.view(dtype=np.int32).reshape(
                                        properties.shape[0], -1
                                    )
                                    for e in self.explode_centers
                                ]
                            ),
                            np.asarray(
                                [[e] * len(properties) for e in self.explode_centers]
                            ).reshape(-1, 1),
                        ]
                    ),
                    dtype=np.int32,
                )

                properties = Labels(list(properties.names) + ["species_center"], pval)
                values, samples, gradient_data, gradient_samples, _ = output

            new_block = TensorBlock(
                values=values,
                samples=Labels(["structure"], samples.reshape(-1, 1)),
                components=block.components,
                properties=properties,
            )

            if gradient_data is not None:
                new_block.add_gradient(
                    "positions",
                    gradient_data,
                    gradient_samples,
                    gradient_components,
                )

            blocks.append(new_block)

        return TensorMap(keys=descriptor.keys, blocks=blocks)


def remove_gradient(tensor):
    blocks = []
    for _, block in tensor:
        blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    return TensorMap(keys=tensor.keys, blocks=blocks)
