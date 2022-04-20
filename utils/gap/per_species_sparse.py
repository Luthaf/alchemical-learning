import numpy as np

import torch
from typing import Dict, List, Optional

from skcosmo.sample_selection import FPS

from .common import CosineKernel


class SparseGapPerSpecies(torch.nn.Module):
    def __init__(
        self,
        power_spectrum: torch.Tensor,
        all_species: torch.Tensor,
        structures_slices: List[slice],
        n_support: Dict[int, int],
        zeta: int,
        energies: torch.Tensor,
        forces: Optional[torch.Tensor] = None,
        lambdas=[1e-12, 1e-12],
        jitter=1e-13,
        optimizable_weights=False,
    ):
        super().__init__()

        if forces is not None:
            raise ValueError("fitting with forces is not implemented")

        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )

        self.n_support = n_support
        self.selected_points = _select_support_points(
            normalized_power_spectrum, all_species, n_support
        )

        kernels = {}
        for species, selected in self.selected_points.items():
            kernels[species] = CosineKernel(
                normalized_power_spectrum[selected], zeta=zeta
            )

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

        weights_per_species = {}
        start = 0
        for species, kernel in kernels.items():
            stop = start + kernel.support_points.shape[0]
            weights_per_species[species] = weights[start:stop, :].T
            start = stop

        self.kernels = torch.nn.ModuleDict({str(s): k for s, k in kernels.items()})

        if optimizable_weights:
            self.weights = torch.nn.ParameterDict(
                {
                    str(s): torch.nn.Parameter(w.detach())
                    for s, w in weights_per_species.items()
                }
            )
        else:
            self.weights = {str(s): w for s, w in weights_per_species.items()}

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

    def update_support_points(
        self,
        power_spectrum,
        all_species,
        structures_slices,
        select_again=False,
    ):
        normalized_power_spectrum = power_spectrum / torch.linalg.norm(
            power_spectrum, dim=1, keepdim=True
        )

        if select_again:
            self.selected_points = _select_support_points(
                normalized_power_spectrum, all_species, self.n_support
            )

        for species, selected in self.selected_points.items():
            self.kernels[str(species)].update_support_points(
                normalized_power_spectrum[selected]
            )


def _select_support_points(power_spectrum, all_species, n_support):
    support_points = {}
    for species, n_to_select in n_support.items():
        X = power_spectrum[all_species == species, :]

        fps = FPS(n_to_select=n_to_select)
        fps.fit(X.detach().cpu().numpy())

        support_points[species] = fps.selected_idx_

    return support_points
