import numpy as np

import torch
from typing import List

from .common import CosineKernel, SumStructureKernel


class FullGapModel(torch.nn.Module):
    def __init__(
        self,
        kernel: CosineKernel,
        training_slices: List[slice],
        weights: torch.Tensor,
        optimize_weights=False,
    ):
        super().__init__()
        self.kernel = kernel
        self.training_slices = training_slices

        if optimize_weights:
            self.weights = torch.nn.Parameter(weights.detach())
        else:
            self.weights = weights

    def forward(self, power_spectrum, all_species, structures_slices):
        assert all_species.shape[0] == power_spectrum.shape[0]

        k_atom_atom = self.kernel(power_spectrum)
        k = SumStructureKernel.apply(
            k_atom_atom, structures_slices, self.training_slices
        )

        return k @ self.weights.T


def train_full_gap_model(
    power_spectrum: torch.Tensor,
    all_species: torch.Tensor,
    structures_slices: List[slice],
    energies,
    zeta=2,
    lambdas=[1e-12, 1e-12],
    optimizable_weights=False,
):
    normalized_power_spectrum = power_spectrum / torch.linalg.norm(
        power_spectrum, dim=1, keepdim=True
    )
    # TODO: we don't necessarily want to detach here, especially for
    # optimizable_weights=True since normalized_power_spectrum can depend on the
    # species coupling explicitly. This will also be the case for sparse models,
    # but it will be even harder since torch can not see through the FPS
    # selection
    kernel = CosineKernel(normalized_power_spectrum.detach(), zeta=zeta)

    k_atom_atom = kernel(power_spectrum)
    K_NN = SumStructureKernel.apply(k_atom_atom, structures_slices, structures_slices)

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

    return FullGapModel(kernel, structures_slices, weights.T, optimizable_weights)
