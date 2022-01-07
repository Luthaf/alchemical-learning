import numpy as np
import math

import torch
from typing import Dict, List

from skcosmo.sample_selection import FPS


class CosineKernel(torch.nn.Module):
    def __init__(self, support_points, zeta=2):
        super().__init__()
        self.register_buffer("support_points", support_points)
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
            self.weights = torch.nn.Parameter(weights)
        else:
            self.weights = weights

    def forward(self, power_spectrum, all_species, structures_slices):
        assert all_species.shape[0] == power_spectrum.shape[0]

        n_structures = len(structures_slices)
        k = torch.zeros(
            (n_structures, len(self.training_slices)),
            device=power_spectrum.device,
        )

        for i, structure_i in enumerate(structures_slices):
            k_atom_atom = self.kernel(power_spectrum[structure_i])
            for j, structure_j in enumerate(self.training_slices):
                k[i, j] = k_atom_atom[:, structure_j].sum()

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
    power_spectrum = power_spectrum.detach()
    normalized_power_spectrum = power_spectrum / torch.linalg.norm(
        power_spectrum, dim=1, keepdim=True
    )
    kernel = CosineKernel(normalized_power_spectrum, zeta=zeta)

    n_structures = len(structures_slices)
    K_NN = torch.zeros(
        (n_structures, n_structures),
        device=power_spectrum.device,
    )
    for i, structure_i in enumerate(structures_slices):
        k_atom_atom = kernel(power_spectrum[structure_i])

        for j, structure_j in enumerate(structures_slices):
            K_NN[i, j] = k_atom_atom[:, structure_j].sum()

    energies = energies.detach().clone().reshape((-1, 1))
    delta = torch.std(energies)

    old_k = K_NN.clone()

    n_atoms_per_frame = torch.tensor([s.stop - s.start for s in structures_slices])
    # regularize the kernel
    K_NN[np.diag_indices_from(K_NN)] += (
        lambdas[0] / delta * torch.sqrt(n_atoms_per_frame)
    )

    weights = torch.linalg.lstsq(K_NN, energies, rcond=None)[0]

    return FullGapModel(kernel, structures_slices, weights.T, optimizable_weights)


class SparseGapModel(torch.nn.Module):
    def __init__(
        self,
        kernels: Dict[int, CosineKernel],
        weights: Dict[int, torch.Tensor],
        optimize_weights=False,
    ):
        super().__init__()
        self.kernels = torch.nn.ModuleDict({str(s): k for s, k in kernels.items()})

        if optimize_weights:
            self.weights = torch.nn.ParameterDict(
                {str(s): torch.nn.Parameter(w) for s, w in weights.items()}
            )
        else:
            self.weights = {str(s): w for s, w in weights.items()}

    def forward(self, power_spectrum, all_species, structures_slices):
        assert all_species.shape[0] == power_spectrum.shape[0]

        energy = torch.zeros((len(structures_slices), 1), device=power_spectrum.device)
        for i, structure in enumerate(structures_slices):
            ps = power_spectrum[structure]
            species = all_species[structure]

            for s, kernel in self.kernels.items():
                k = kernel(ps[species == int(s), :]).sum(dim=0)
                energy[i] += k @ self.weights[str(s)].T

        return energy


def train_sparse_gap_model(
    power_spectrum: torch.Tensor,
    all_species: torch.Tensor,
    structures_slices: List[slice],
    energies,
    n_support,
    zeta=2,
    lambdas=[1e-12, 1e-12],
    jitter=1e-13,
    optimizable_weights=False,
):
    power_spectrum = power_spectrum.detach()
    support_points = select_support_points(power_spectrum, all_species, n_support)

    kernels = {}
    for species, support in support_points.items():
        kernels[species] = CosineKernel(support, zeta=zeta)

    K_MM = torch.block_diag(*[kernel.compute_KMM() for kernel in kernels.values()])

    K_NM_per_species = []
    for s, kernel in kernels.items():
        K_NM_per_frame = []
        for structure in structures_slices:
            ps = power_spectrum[structure]
            species = all_species[structure]

            K_NM_per_frame.append(kernel(ps[species == s, :]).sum(dim=0))

        K_NM_per_species.append(torch.vstack(K_NM_per_frame))

    K_NM = torch.hstack(K_NM_per_species)

    # finish building the kernel
    energies = energies.detach().clone().reshape((-1, 1))
    delta = torch.std(energies)

    n_atoms_per_frame = torch.tensor([s.stop - s.start for s in structures_slices])
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
        weights_per_species[species] = (
            weights[start:stop, :].detach().T.clone().contiguous()
        )
        start = stop

    return SparseGapModel(kernels, weights_per_species, optimizable_weights)


def select_support_points(power_spectrum, all_species, n_support):
    features = power_spectrum / torch.linalg.norm(power_spectrum, dim=1, keepdim=True)

    support_points = {}
    for species, n_to_select in n_support.items():
        X = features[all_species == species, :]

        fps = FPS(n_to_select=n_to_select)
        selected = fps.fit_transform(X.detach().numpy())

        support_points[species] = torch.tensor(selected)

    return support_points
