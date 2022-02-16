import numpy as np

import torch
from typing import List

from .common import CosineKernel, SumStructureKernel


class FullGap(torch.nn.Module):
    def __init__(
        self,
        power_spectrum: torch.Tensor,
        structures_slices: List[slice],
        energies: torch.Tensor,
        zeta: int,
        lambdas=[1e-12, 1e-12],
        optimizable_weights=False,
    ):
        super().__init__()
        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )
        self.kernel = CosineKernel(normalized_power_spectrum, zeta=zeta)

        k_atom_atom = self.kernel(power_spectrum)
        K_NN = SumStructureKernel.apply(
            k_atom_atom, structures_slices, structures_slices
        )

        energies = energies.detach().clone().reshape((-1, 1))
        delta = torch.std(energies)

        n_atoms_per_frame = torch.tensor(
            [s.stop - s.start for s in structures_slices],
            device=power_spectrum.device,
        )
        # regularize the kernel
        K_NN[np.diag_indices_from(K_NN)] += (
            lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)
        )
        weights = torch.linalg.solve(K_NN, energies)

        self.training_slices = structures_slices

        if optimizable_weights:
            self.weights = torch.nn.Parameter(weights.T.detach())
        else:
            self.weights = weights.T

    def update_support_points(self, power_spectrum, all_species, select_again=False):
        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )
        self.kernel.update_support_points(normalized_power_spectrum)

    def forward(self, power_spectrum, all_species, structures_slices):
        k_atom_atom = self.kernel(power_spectrum)
        k = SumStructureKernel.apply(
            k_atom_atom, structures_slices, self.training_slices
        )

        return k @ self.weights.T
