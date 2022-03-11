import numpy as np

import torch
from typing import List

from skcosmo.sample_selection import FPS

from .common import CosineKernel


class SparseGap(torch.nn.Module):
    def __init__(
        self,
        power_spectrum: torch.Tensor,
        structures_slices: List[slice],
        energies: torch.Tensor,
        n_support: int,
        zeta=2,
        lambdas=[1e-12, 1e-12],
        jitter=1e-13,
        optimizable_weights=False,
        detach_support_points=False,
    ):
        super().__init__()

        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )

        self.n_support = n_support
        self.selected_points = _select_support_points(
            normalized_power_spectrum, n_support
        )

        support_points = normalized_power_spectrum[self.selected_points]
        self.kernel = CosineKernel(
            support_points,
            zeta=zeta,
            detach_support_points=detach_support_points,
        )

        K_MM = self.kernel.compute_KMM()

        K_NM_per_frame = []
        for structure in structures_slices:
            K_NM_per_frame.append(self.kernel(power_spectrum[structure]).sum(dim=0))

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

        if optimizable_weights:
            self.weights = torch.nn.Parameter(weights.detach())
        else:
            self.weights = weights

    def update_support_points(self, power_spectrum, all_species, select_again=False):
        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )

        if select_again:
            self.selected_points = _select_support_points(
                normalized_power_spectrum, self.n_support
            )

        self.kernel.update_support_points(
            normalized_power_spectrum[self.selected_points]
        )

    def forward(self, power_spectrum, all_species, structures_slices):
        energy = torch.zeros((len(structures_slices), 1), device=power_spectrum.device)
        for i, structure in enumerate(structures_slices):
            ps = power_spectrum[structure]

            k = self.kernel(ps).sum(dim=0)
            energy[i] += k @ self.weights

        return energy


def _select_support_points(power_spectrum, n_support):
    fps = FPS(n_to_select=n_support)
    fps.fit(power_spectrum.detach().cpu().numpy())
    return fps.selected_idx_
