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
                blocks.append(
                    TensorBlock(
                        values=torch.tensor(block.values[mask].copy()),
                        samples=block.samples[mask],
                        components=block.components,
                        properties=block.properties,
                    )
                )

                if block.has_gradient("positions"):
                    raise ValueError("unimplemented: Dataset with gradients")

            self.spherical_expansions.append(
                TensorMap(spherical_expansions.keys, blocks)
            )

        self.frames = frames
        self.energies = energies

        if forces is not None:
            raise ValueError("not implemented: dataset with forces")

    def __len__(self):
        return len(self.spherical_expansions)

    def __getitem__(self, idx):
        return (self.spherical_expansions[idx], self.energies[idx])


def _collate_data(device):
    def do_collate(data):
        keys = data[0][0].keys

        blocks = []
        for key in keys:
            first_block = data[0][0].block(key)
            samples = []
            values = []
            for (spx, _) in data:
                block = spx.block(key)
                samples.append(block.samples.view(dtype=np.int32).reshape(-1, 3))
                values.append(block.values)

                if block.has_gradient("positions"):
                    raise ValueError("unimplemented: collate_data with gradients")

            blocks.append(
                TensorBlock(
                    values=torch.vstack(values).to(device),
                    samples=Labels(
                        ["structure", "center", "species_center"], np.vstack(samples)
                    ),
                    components=first_block.components,
                    properties=first_block.properties,
                )
            )

        spherical_expansion = TensorMap(keys, blocks)
        energies = torch.vstack([d[1] for d in data])

        return spherical_expansion, energies

    return do_collate


def create_dataloader(dataset, batch_size, shuffle=True, device="cpu"):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_data(device),
    )
