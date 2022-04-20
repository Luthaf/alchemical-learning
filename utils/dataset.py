import torch

from .soap import TorchFrame, compute_spherical_expansion_librascal


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(self, frames, hypers, energies, positions_requires_grad=False):
        self.spherical_expansions = []
        self.frames = [
            TorchFrame(frame, requires_grad=positions_requires_grad) for frame in frames
        ]
        for frame in self.frames:
            se, slices = compute_spherical_expansion_librascal(
                [frame], hypers, gradients=positions_requires_grad
            )
            self.spherical_expansions.append(se)

        self.energies = energies

    def __len__(self):
        return len(self.spherical_expansions)

    def __getitem__(self, idx):
        return (
            self.spherical_expansions[idx],
            self.frames[idx].species,
            self.energies[idx],
        )

    def positions_grad(self):
        return torch.vstack([frame.positions.grad for frame in self.frames])

    def zero_positions_grad(self):
        for frame in self.frames:
            frame.positions.grad = None


def _collate_data_cpu(data):
    spherical_expansion = {
        lambda_: torch.vstack([d[0][lambda_] for d in data])
        for lambda_ in data[0][0].keys()
    }

    species = torch.hstack([d[1] for d in data])
    energies = torch.vstack([d[2] for d in data])

    slices = []
    start = 0
    for d in data:
        stop = start + d[1].shape[0]
        slices.append(slice(start, stop))
        start = stop

    return spherical_expansion, species, slices, energies


def _collate_data_gpu(data):
    spherical_expansion, species, slices, energies = _collate_data_cpu(data)

    spherical_expansion = {
        lambda_: se.to(device="cuda") for lambda_, se in spherical_expansion.items()
    }

    return (
        spherical_expansion,
        species.to(device="cuda"),
        slices,
        energies.to(device="cuda"),
    )


def create_dataloader(dataset, batch_size, shuffle=True, device="cpu"):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_data_gpu if device == "cuda" else _collate_data_cpu,
    )
