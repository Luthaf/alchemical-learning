import math
import torch
from typing import Dict

from rascal.representations import SphericalExpansion


def compute_spherical_expansion_librascal(frames, hypers):
    calculator = SphericalExpansion(**hypers)

    max_radial = hypers["max_radial"]
    max_angular = hypers["max_angular"]
    n_species = len(hypers["global_species"])

    result = []
    for frame in frames:
        manager = calculator.transform(frame)
        se = manager.get_features(calculator)

        # Transform from [i_center, alpha, n, lm] to [i_center, lm, alpha, n]
        se = se.reshape(se.shape[0], n_species * max_radial, -1)
        se = se.swapaxes(1, 2)

        spherical_expansion = {}
        start = 0
        for l in range(max_angular + 1):
            stop = start + 2 * l + 1
            spherical_expansion[l] = torch.tensor(se[:, start:stop, :])
            start = stop

        result.append(spherical_expansion)

    return result


class PowerSpectrum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spherical_expansions: Dict[int, torch.Tensor]):
        n_environments = spherical_expansions[0].shape[0]
        n_angular = len(spherical_expansions)
        n_species_radial = spherical_expansions[0].shape[2]
        feature_size = n_angular * n_species_radial * n_species_radial

        output = torch.zeros((n_environments, feature_size))
        for l, spherical_expansion in spherical_expansions.items():
            start = l * n_species_radial * n_species_radial
            stop = (l + 1) * n_species_radial * n_species_radial

            power_spectrum = torch.einsum(
                "i m q, i m r -> i m q r", spherical_expansion, spherical_expansion
            )
            power_spectrum = power_spectrum.reshape(
                spherical_expansion.shape[0], spherical_expansion.shape[1], -1
            )
            power_spectrum = power_spectrum.sum(dim=1) / math.sqrt(2 * l + 1)

            output[:, start:stop] = power_spectrum

        return output
