import torch
from typing import Optional
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


class SumStructuresAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        samples: Labels,
        gradient_data: Optional[torch.Tensor],
        gradient_samples: Optional[Labels],
    ):
        # get the unique entries in samples["structure"] without
        # sorting the result (that would break pytorch dataloader shuffling)
        unique_structures_idx = np.unique(samples["structure"], return_index=True)[1]
        new_samples = samples["structure"][np.sort(unique_structures_idx)]

        new_values = torch.zeros(
            (len(new_samples), *values.shape[1:]),
            device=values.device,
        )

        if gradient_data is not None:
            assert gradient_samples is not None
            do_gradients = True

            cumulative_atoms_per_frame = [0]
            for structure_i, structure in enumerate(new_samples):
                mask = gradient_samples["structure"] == structure
                atoms = np.unique(gradient_samples[mask]["atom"])

                cumulative_atoms = len(atoms) + cumulative_atoms_per_frame[-1]
                cumulative_atoms_per_frame.append(cumulative_atoms)

            new_gradient_data = torch.zeros(
                (cumulative_atoms_per_frame[-1], *gradient_data.shape[1:]),
                device=gradient_data.device,
            )

            new_gradient_samples = []
            grad_samples_mapping = [[] for _ in range(new_gradient_data.shape[0])]
        else:
            do_gradients = False
            new_gradient_data = None
            new_gradient_samples = None
            grad_samples_mapping = None

        structures_masks = []
        for structure_i, structure in enumerate(new_samples):
            mask = samples["structure"] == structure
            structures_masks.append(mask)
            new_values[structure_i, ...] = values[mask, ...].sum(dim=0, keepdim=True)

            if do_gradients:
                mask = gradient_samples["structure"] == structure
                atoms = np.unique(gradient_samples[mask]["atom"])

                new_gradient_samples.append(
                    np.array(
                        [[structure_i, structure, atom] for atom in atoms],
                        dtype=np.int32,
                    )
                )

                atom_index_positions = {atom: i for i, atom in enumerate(atoms)}

                for sample_i in np.where(gradient_samples["structure"] == structure)[0]:
                    grad_sample = gradient_samples[sample_i]

                    atom_i = atom_index_positions[grad_sample["atom"]]
                    grad_sample_i = cumulative_atoms_per_frame[structure_i] + atom_i
                    new_gradient_data[grad_sample_i] += gradient_data[sample_i]
                    grad_samples_mapping[grad_sample_i].append(sample_i)

        if do_gradients:
            new_gradient_samples = Labels(
                names=["sample", "structure", "atom"],
                values=np.concatenate(new_gradient_samples),
            )

        ctx.structures_masks = structures_masks
        ctx.grad_samples_mapping = grad_samples_mapping
        ctx.save_for_backward(values, gradient_data)

        return new_values, new_samples, new_gradient_data, new_gradient_samples

    @staticmethod
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
            grad_values = torch.zeros_like(values)
            for structure_i, mask in enumerate(ctx.structures_masks):
                grad_values[mask, ...] = grad_new_values[structure_i, ...]

        if gradient_data is not None and gradient_data.requires_grad:
            grad_gradient_data = torch.zeros_like(gradient_data)
            for sample_i, mask in enumerate(ctx.grad_samples_mapping):
                grad_gradient_data[mask, ...] = grad_new_gradient_data[sample_i, ...]

        return grad_values, None, grad_gradient_data, None


class SumStructures(torch.nn.Module):
    def __init__(self, sum_properties=False):
        super().__init__()
        self.sum_properties = sum_properties
        if self.sum_properties:
            raise NotImplementedError()

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

            output = SumStructuresAutograd.apply(
                block.values,
                block.samples,
                gradient_data,
                gradient_samples,
            )
            values, samples, gradient_data, gradient_samples = output

            new_block = TensorBlock(
                values=values,
                samples=Labels(["structure"], samples.reshape(-1, 1)),
                components=block.components,
                properties=block.properties,
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
