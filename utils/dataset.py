import torch
import numpy as np

from equistore import TensorBlock, TensorMap, Labels
from .rascaline import RascalineSphericalExpansion


def _move_to_torch(tensor_map, structure_i):
    blocks = []
    for _, block in tensor_map:
        assert block.samples.names == ("structure", "center", "center_species")
        samples = block.samples.view(dtype=np.int32).reshape(-1, 3).copy()
        samples[:, 0] = structure_i
        samples = Labels(block.samples.names, samples)

        new_block = TensorBlock(
            values=torch.tensor(block.values),
            samples=samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)

            assert gradient.samples.names == ("sample", "structure", "atom")

            gradient_samples = (
                gradient.samples.view(dtype=np.int32).reshape(-1, 3).copy()
            )
            gradient_samples[:, 1] = structure_i
            gradient_samples = Labels(gradient.samples.names, gradient_samples)

            new_block.add_gradient(
                parameter=parameter,
                data=torch.tensor(gradient.data),
                samples=gradient_samples,
                components=gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(tensor_map.keys, blocks)


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(self, frames, all_species, hypers, energies, forces=None):
        all_species = Labels(
            names=["neighbor_species"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )

        calculator = RascalineSphericalExpansion(hypers)

        self.spherical_expansions = []
        for frame_i, frame in enumerate(frames):
            spherical_expansion = calculator.compute(frame)
            spherical_expansion.keys_to_samples("center_species")
            spherical_expansion.keys_to_properties(all_species)

            self.spherical_expansions.append(
                _move_to_torch(spherical_expansion, frame_i)
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
        energies = torch.vstack([d[1] for d in data]).to(device=device)

        if data[0][2] is not None:
            forces = torch.vstack([d[2] for d in data]).to(device=device)
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
