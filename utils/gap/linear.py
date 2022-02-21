import numpy as np

import torch
from typing import List


class SumStructures(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        structure_slices: List[slice],
    ):
        output = torch.zeros(
            (len(structure_slices), features.shape[1]),
            device=features.device,
        )
        for i, structure_i in enumerate(structure_slices):
            output[i, :] = features[structure_i, :].sum(dim=0, keepdim=True)

        ctx.structure_slices = structure_slices
        ctx.save_for_backward(features)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        kernel = ctx.saved_tensors[0]

        if kernel.requires_grad:
            grad_input = torch.zeros_like(kernel)
            for i, structure_i in enumerate(ctx.structure_slices):
                grad_input[structure_i, :] = grad_output[i, :]

        return grad_input, None, None


class RidgeRegression(torch.nn.Module):
    def __init__(
        self,
        power_spectrum: torch.Tensor,
        structures_slices: List[slice],
        energies: torch.Tensor,
        lambdas=[1e-12, 1e-12],
        optimizable_weights=False,
    ):
        super().__init__()

        structure_power_spectrum = SumStructures.apply(
            power_spectrum, structures_slices
        )

        XT_X = structure_power_spectrum.T @ structure_power_spectrum

        energies = energies.detach().clone().reshape((-1, 1))
        delta = torch.std(energies)

        # regularize
        XT_X[np.diag_indices_from(XT_X)] += lambdas[0] / delta

        weights = torch.linalg.solve(XT_X, structure_power_spectrum.T @ energies)

        if optimizable_weights:
            self.weights = torch.nn.Parameter(weights.T.detach())
        else:
            self.weights = weights.T

    def update_support_points(self, power_spectrum, all_species, select_again=False):
        return

    def forward(self, power_spectrum, all_species, structures_slices):
        X = SumStructures.apply(power_spectrum, structures_slices)

        return X @ self.weights.T
