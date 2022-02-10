import numpy as np

import torch
from typing import Dict, List

from skcosmo.sample_selection import FPS

from .common import CosineKernel


class SparseGapModel(torch.nn.Module):
    def __init__(
        self,
        kernel: CosineKernel,
        weights: torch.Tensor,
        optimize_weights=False,
    ):
        super().__init__()
        self.kernel = kernel

        if optimize_weights:
            self.weights = torch.nn.Parameter(weights.detach())
        else:
            self.weights = weights.detach()

    def forward(self, power_spectrum, all_species, structures_slices):
        assert all_species.shape[0] == power_spectrum.shape[0]

        energy = torch.zeros((len(structures_slices), 1), device=power_spectrum.device)
        for i, structure in enumerate(structures_slices):
            ps = power_spectrum[structure]

            k = self.kernel(ps).sum(dim=0)
            energy[i] += k @ self.weights

        return energy


def train_sparse_gap_model(
    power_spectrum: torch.Tensor,
    all_species: torch.Tensor,
    structures_slices: List[slice],
    energies: torch.Tensor,
    n_support: Dict[int, int],
    zeta=2,
    lambdas=[1e-12, 1e-12],
    jitter=1e-13,
    optimizable_weights=False,
):
    support_points = _select_support_points(power_spectrum, n_support)
    kernel = CosineKernel(support_points, zeta=zeta)

    K_MM = kernel.compute_KMM()

    K_NM_per_frame = []
    for structure in structures_slices:
        K_NM_per_frame.append(kernel(power_spectrum[structure]).sum(dim=0))

    K_NM = torch.vstack(K_NM_per_frame)

    # finish building the kernel
    energies = energies.detach().clone().reshape((-1, 1))
    delta = torch.std(energies)

    n_atoms_per_frame = torch.tensor(
        [s.stop - s.start for s in structures_slices],
        device=power_spectrum.device,
    )
    K_NM[:] /= lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)[:, None]
    energies /= lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)[:, None]

    Y = energies
    K_MM[np.diag_indices_from(K_MM)] += jitter

    K = K_MM + K_NM.T @ K_NM
    Y = K_NM.T @ Y

    weights = torch.linalg.solve(K, Y)

    return SparseGapModel(kernel, weights, optimizable_weights)


def _select_support_points(power_spectrum, n_support):
    features = power_spectrum / torch.linalg.norm(power_spectrum, dim=1, keepdim=True)

    fps = FPS(n_to_select=n_support)
    fps.fit(features.detach().cpu().numpy())

    return features[fps.selected_idx_]
