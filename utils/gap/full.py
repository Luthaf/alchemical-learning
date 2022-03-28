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
        random_initial_weights=False,
        detach_support_points=False,
    ):
        super().__init__()

        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )

        normalized_power_spectrum = normalized_power_spectrum.detach()

        self.kernel = CosineKernel(
            normalized_power_spectrum,
            zeta=zeta,
            detach_support_points=detach_support_points,
        )

        if random_initial_weights:
            weights = torch.rand(
                (1, len(structures_slices)),
                device=power_spectrum.device,
            )
        else:
            k_atom_atom = self.kernel(power_spectrum)
            weights = _fit_full_kernel(
                k_atom_atom, structures_slices, energies, lambdas
            )

        if optimizable_weights:
            self.weights = torch.nn.Parameter(weights.detach())
        else:
            self.weights = weights

        self.training_slices = structures_slices

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


def _fit_full_kernel(k_atom_atom, structures_slices, energies, lambdas):
    K_NN = SumStructureKernel.apply(k_atom_atom, structures_slices, structures_slices)

    energies = energies.detach().clone().reshape((-1, 1))
    delta = torch.std(energies)

    n_atoms_per_frame = torch.tensor(
        [s.stop - s.start for s in structures_slices],
        device=k_atom_atom.device,
    )
    # regularize the kernel
    K_NN[np.diag_indices_from(K_NN)] += (
        lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)
    )
    weights = torch.linalg.solve(K_NN, energies)

    return weights.T
