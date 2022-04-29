import torch
import numpy as np

from equistore import TensorBlock, TensorMap, Labels
from .rascaline import RascalineSphericalExpansion


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(self, frames, hypers, energies, forces=None):
        calculator = RascalineSphericalExpansion(hypers)
        spherical_expansions = calculator.compute(frames)

        spherical_expansions.keys_to_samples("center_species")
        spherical_expansions.keys_to_properties("neighbor_species")

        self.spherical_expansions = []
        for structure_i in range(len(frames)):
            blocks = []
            for _, block in spherical_expansions:
                mask = block.samples["structure"] == structure_i

                new_block = TensorBlock(
                    values=torch.tensor(block.values[mask].copy()),
                    samples=block.samples[mask],
                    components=block.components,
                    properties=block.properties,
                )

                if block.has_gradient("positions"):
                    gradient = block.gradient("positions")

                    new_grad_samples = []
                    new_grad_data = []
                    for sample_i, sample in enumerate(np.where(mask)[0]):
                        grad_mask = gradient.samples["sample"] == sample

                        new_grad_data.append(
                            torch.tensor(gradient.data[grad_mask].copy())
                        )

                        # update the sample id in gradients to only span the
                        # current structure
                        new_samples = (
                            gradient.samples[grad_mask]
                            .copy()
                            .view(dtype=np.int32)
                            .reshape(-1, 3)
                        )
                        new_samples[:, 0] = sample_i
                        new_grad_samples.append(new_samples)

                    new_block.add_gradient(
                        "positions",
                        data=torch.vstack(new_grad_data),
                        samples=Labels(
                            ["sample", "structure", "atom"],
                            np.vstack(new_grad_samples),
                        ),
                        components=gradient.components,
                    )

                blocks.append(new_block)

            self.spherical_expansions.append(
                TensorMap(spherical_expansions.keys, blocks)
            )

        self.frames = frames

        assert isinstance(energies, torch.Tensor)
        assert energies.shape == (len(frames), 1)
        self.energies = energies

        if forces is not None:
            assert isinstance(forces, list)
            for i, f in enumerate(forces):
                assert isinstance(f, torch.Tensor)
                assert f.shape == (len(self.frames[i]), 3)

        self.forces = forces

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if self.forces is None:
            forces = None
        else:
            forces = self.forces[idx]

        return (self.spherical_expansions[idx], self.energies[idx], forces)


def _collate_data(device):
    def do_collate(data):
        keys = data[0][0].keys

        blocks = []
        for key in keys:
            first_block = data[0][0].block(key)
            if first_block.has_gradient("positions"):
                first_block_grad = first_block.gradient("positions")
            else:
                first_block_grad = None

            samples = []
            values = []

            grad_samples = []
            grad_data = []
            previous_samples_count = 0
            for spx, _, _ in data:
                block = spx.block(key)

                new_samples = block.samples.view(dtype=np.int32).reshape(-1, 3)
                samples.append(new_samples)
                values.append(block.values)

                if block.has_gradient("positions"):
                    gradient = block.gradient("positions")

                    new_grad_samples = (
                        gradient.samples.view(dtype=np.int32).reshape(-1, 3).copy()
                    )
                    new_grad_samples[:, 0] += previous_samples_count
                    grad_samples.append(new_grad_samples)

                    grad_data.append(gradient.data)

                previous_samples_count += new_samples.shape[0]

            # print(samples)
            # raise 44
            new_block = TensorBlock(
                values=torch.vstack(values).to(device),
                samples=Labels(
                    ["structure", "center", "species_center"],
                    np.vstack(samples),
                ),
                components=first_block.components,
                properties=first_block.properties,
            )
            if first_block_grad is not None:
                new_block.add_gradient(
                    "positions",
                    data=torch.vstack(grad_data).to(device),
                    samples=Labels(
                        ["sample", "structure", "atom"],
                        np.vstack(grad_samples),
                    ),
                    components=first_block_grad.components,
                )

            blocks.append(new_block)

        spherical_expansion = TensorMap(keys, blocks)
        energies = torch.vstack([d[1] for d in data])

        if data[0][2] is not None:
            forces = torch.vstack([d[2] for d in data])
        else:
            forces = None

        return spherical_expansion, energies, forces

    return do_collate


def create_dataloader(dataset, batch_size, shuffle=True, device="cpu"):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_data(device),
    )
