import torch
import numpy as np

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

            gradient_data = gradient.data / norm[gradient.samples["sample"], None, None]

            # gradient of x_i = X_i / N_i is given by
            # 1 / N_i \grad X_i - x_i [x_i @ 1 / N_i \grad X_i]
            for sample_i, (sample, _, _) in enumerate(gradient.samples):
                dot = gradient_data[sample_i] @ normalized_values[sample].T

                gradient_data[sample_i, 0, :] -= dot[0] * normalized_values[sample, :]
                gradient_data[sample_i, 1, :] -= dot[1] * normalized_values[sample, :]
                gradient_data[sample_i, 2, :] -= dot[2] * normalized_values[sample, :]

            new_block.add_gradient(
                "positions", gradient_data, gradient.samples, gradient.components
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


class SumStructures(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        samples: Labels,
    ):
        # get the unique entries in samples["structure"] without
        # sorting the result (that would break pytorch dataloader shuffling)
        unique_structures_idx = np.unique(samples["structure"], return_index=True)[1]
        structures = samples["structure"][np.sort(unique_structures_idx)]

        output = torch.zeros(
            (len(structures), values.shape[1]),
            device=values.device,
        )

        structures_masks = []
        for structure_i, structure in enumerate(structures):
            mask = samples["structure"] == structure
            structures_masks.append(mask)
            output[structure_i, :] = values[mask, :].sum(dim=0, keepdim=True)

        ctx.structures_masks = structures_masks
        ctx.save_for_backward(values)

        return output, structures

    @staticmethod
    def backward(ctx, grad_output, grad_structures):
        grad_input = None
        values = ctx.saved_tensors[0]

        if values.requires_grad:
            grad_input = torch.zeros_like(values)
            for structure_i, mask in enumerate(ctx.structures_masks):
                grad_input[mask, :] = grad_output[structure_i, :]

        return grad_input, None


def structure_sum(descriptor, sum_properties=False):
    if sum_properties:
        raise ValueError("not implemented")

    blocks = []
    for _, block in descriptor:
        # no lambda kernels for now
        assert len(block.components) == 0

        summed_values, structures = SumStructures.apply(block.values, block.samples)

        new_block = TensorBlock(
            values=summed_values,
            samples=Labels(["structure"], structures.reshape(-1, 1)),
            components=block.components,
            properties=block.properties,
        )

        blocks.append(new_block)

    return TensorMap(keys=descriptor.keys, blocks=blocks)
