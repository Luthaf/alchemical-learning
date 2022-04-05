from typing import Dict
import torch
import math
import copy

import ase

from rascal.representations import SphericalExpansion


def compute_spherical_expansion_librascal(frames, hypers, gradients=False):
    hypers = copy.deepcopy(hypers)
    hypers["compute_gradients"] = gradients
    calculator = SphericalExpansion(**hypers)

    se = []
    for frame in frames:
        assert isinstance(frame, TorchFrame)
        se.append(
            SphericalExpansionAutograd.apply(
                calculator,
                frame.positions,
                frame.cell,
                frame.species,
            )
        )
    se = torch.vstack(se)

    max_radial = hypers["max_radial"]
    max_angular = hypers["max_angular"]
    n_species = len(hypers["global_species"])

    # Transform from [i_center, alpha, n, lm] to [i_center, lm, alpha, n]
    se = se.reshape(se.shape[0], n_species * max_radial, -1)
    se = se.swapaxes(1, 2)

    spherical_expansion = {}
    start = 0
    for l in range(max_angular + 1):
        stop = start + 2 * l + 1
        spherical_expansion[l] = se[:, start:stop, :]
        start = stop

    structures_slices = []
    n_atoms_before = 0
    for frame in frames:
        structures_slices.append(slice(n_atoms_before, n_atoms_before + len(frame)))
        n_atoms_before += len(frame)

    return spherical_expansion, structures_slices


class PowerSpectrum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spherical_expansions: Dict[int, torch.Tensor]):
        n_environments = spherical_expansions[0].shape[0]
        n_angular = len(spherical_expansions)
        n_species_radial = spherical_expansions[0].shape[2]
        feature_size = n_angular * n_species_radial * n_species_radial

        output = torch.zeros(
            (n_environments, feature_size), device=spherical_expansions[0].device
        )
        for l, spherical_expansion in spherical_expansions.items():
            start = l * n_species_radial * n_species_radial
            stop = (l + 1) * n_species_radial * n_species_radial

            power_spectrum = torch.einsum(
                "i m q, i m r -> i q r", spherical_expansion, spherical_expansion
            ) / math.sqrt(2 * l + 1)

            power_spectrum = power_spectrum.reshape(spherical_expansion.shape[0], -1)

            output[:, start:stop] = power_spectrum

        return output


class SphericalExpansionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, calculator, positions, cell, numbers):
        assert isinstance(positions, torch.Tensor)
        assert isinstance(cell, torch.Tensor)
        assert isinstance(numbers, torch.Tensor)

        frame = ase.Atoms(
            numbers=numbers.numpy(), cell=cell.numpy(), positions=positions.numpy()
        )
        manager = calculator.transform(frame)
        descriptor = manager.get_features(calculator)

        grad_descriptor = manager.get_features_gradient(calculator)
        grad_info = manager.get_gradients_info()
        ctx.save_for_backward(
            torch.tensor(grad_descriptor),
            torch.tensor(grad_info),
        )

        return torch.tensor(descriptor)

    @staticmethod
    def backward(ctx, grad_output):
        grad_spherical_expansion, grad_info = ctx.saved_tensors

        grad_calculator = grad_positions = grad_cell = grad_numbers = None

        if ctx.needs_input_grad[0]:
            raise ValueError("can not compute gradients w.r.t. calculator")
        if ctx.needs_input_grad[1]:
            if grad_spherical_expansion.shape[1] == 0:
                raise ValueError(
                    "missing gradients, please set `compute_gradients` to True"
                )

            grad_positions = torch.zeros((grad_output.shape[0], 3))
            for sample, (_, center_i, neighbor_i, *_) in enumerate(grad_info):
                for spatial_i in range(3):
                    sample_i = 3 * sample + spatial_i
                    grad_positions[neighbor_i, spatial_i] += torch.dot(
                        grad_output[center_i, :], grad_spherical_expansion[sample_i, :]
                    )

        if ctx.needs_input_grad[2]:
            raise ValueError("can not compute gradients w.r.t. cell")
        if ctx.needs_input_grad[2]:
            raise ValueError("can not compute gradients w.r.t. atomic numbers")

        return grad_calculator, grad_positions, grad_cell, grad_numbers


class TorchFrame:
    def __init__(self, frame: ase.Atoms, requires_grad: bool):
        self.positions = torch.tensor(frame.positions, requires_grad=requires_grad)
        self.species = torch.tensor(frame.numbers)
        self.cell = torch.tensor(frame.cell[:])

    def __len__(self):
        return self.species.shape[0]
