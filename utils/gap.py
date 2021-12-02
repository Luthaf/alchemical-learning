import numpy as np

import torch
from typing import Dict, List

from skcosmo.sample_selection import FPS


class CosineKernel(torch.nn.Module):
    def __init__(self, support_points, zeta=2):
        super().__init__()
        self.support_points = support_points
        assert torch.allclose(
            torch.linalg.norm(support_points, dim=1), torch.tensor(1.0)
        )

        self.zeta = zeta

    def forward(self, power_spectrum: torch.Tensor):
        """Compute K_NM kernel between passed environments and support points"""
        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )
        return torch.pow(normalized_power_spectrum @ self.support_points.T, self.zeta)

    def compute_KMM(self):
        return torch.pow(self.support_points @ self.support_points.T, self.zeta)


class GapModel(torch.nn.Module):
    def __init__(
        self,
        kernels: Dict[int, CosineKernel],
        weights: Dict[int, torch.Tensor],
    ):
        super().__init__()
        self.kernels = kernels
        self.weights = weights

    def forward(self, power_spectrum, all_species):
        assert all_species.shape[0] == power_spectrum.shape[0]

        energy = torch.tensor([0.0])
        for species in self.kernels.keys():
            kernel = self.kernels[species](
                power_spectrum[all_species == species, :]
            ).sum(dim=0)

            energy += kernel @ self.weights[species].T

        return energy


def train_gap_model(
    power_spectrum: List[torch.Tensor],
    all_species,
    energies,
    n_support,
    zeta=2,
    lambdas=[1e-12, 1e-12],
    jitter=1e-13,
):
    energies = energies.clone().reshape((-1, 1))

    support_points = select_support_points(
        torch.vstack(power_spectrum),
        torch.hstack(all_species),
        n_support,
    )

    kernels = {}
    for species, support in support_points.items():
        kernels[species] = CosineKernel(support, zeta=zeta)

    K_MM = torch.block_diag(*[kernel.compute_KMM() for kernel in kernels.values()])

    K_NM_per_species = []
    for s, kernel in kernels.items():
        K_NM_per_frame = []
        for ps, species in zip(power_spectrum, all_species):
            K_NM_per_frame.append(kernel(ps[species == s, :]).sum(dim=0))

        K_NM_per_species.append(torch.vstack(K_NM_per_frame))

    K_NM = torch.hstack(K_NM_per_species)

    # finish building the kernel
    delta = torch.std(energies)

    n_atoms_per_frame = torch.tensor([ps.shape[0] for ps in power_spectrum])
    K_NM[:] /= lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)[:, None]
    energies /= lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)[:, None]

    Y = energies
    K_MM[np.diag_indices_from(K_MM)] += jitter

    K = K_MM + K_NM.T @ K_NM
    Y = K_NM.T @ Y

    weights = torch.linalg.lstsq(K, Y, rcond=None)[0]

    weights_per_species = {}
    start = 0
    for species, kernel in kernels.items():
        stop = start + kernel.support_points.shape[0]
        weights_per_species[species] = weights[start:stop, :].detach().T.clone()
        start = stop

    return GapModel(kernels, weights_per_species)


def select_support_points(power_spectrum, all_species, n_support):
    features = power_spectrum / torch.linalg.norm(power_spectrum, dim=1, keepdim=True)

    support_points = {}
    for species, n_to_select in n_support.items():
        X = features[all_species == species, :]

        fps = FPS(n_to_select=n_to_select)
        selected = fps.fit_transform(X.detach().numpy())

        support_points[species] = torch.tensor(selected)

    return support_points
