import torch
import numpy as np
from typing import Dict

from collections import namedtuple
import sklearn.decomposition


class AlchemicalCombine(torch.nn.Module):
    def __init__(self, species, n_pseudo_species):
        super().__init__()
        coupling = _species_coupling_matrix(species)
        pca = sklearn.decomposition.PCA(n_components=n_pseudo_species)
        self.combining_matrix = torch.nn.Parameter(
            torch.tensor(pca.fit_transform(coupling)).contiguous()
        )

    def forward(self, spherical_expansion: Dict[int, torch.Tensor]):
        n_species = self.combining_matrix.shape[0]

        output = {}
        for l, se in spherical_expansion.items():
            se = se.reshape(se.shape[0], se.shape[1], n_species, -1)

            se = se.swapaxes(2, 3)
            combined = se @ self.combining_matrix
            combined = combined.swapaxes(2, 3)

            output[l] = combined.reshape(se.shape[0], se.shape[1], -1)

        return output


def _species_coupling_matrix(species):
    K = np.zeros((len(species), len(species)))

    SIGMA_EPSILON = 1.0
    SIGMA_RADIUS = 1.0

    for i, species_i in enumerate(species):
        constants_i = ATOMIC_DATA_PER_SPECIES[species_i]
        for j, species_j in enumerate(species):
            constants_j = ATOMIC_DATA_PER_SPECIES[species_j]

            delta_epsilon = (
                constants_i.electronegativity - constants_j.electronegativity
            )
            a = delta_epsilon ** 2 / (2 * SIGMA_EPSILON ** 2)

            delta_radius = constants_i.radius - constants_j.radius
            b = delta_radius ** 2 / (2 * SIGMA_RADIUS ** 2)

            K[i, j] = np.exp(-(a + b))

    return K


AtomicData = namedtuple("AtomicData", ["symbol", "electronegativity", "radius"])

ATOMIC_DATA_PER_SPECIES = {
    1: AtomicData("H", 2.1, 120),
    2: AtomicData("He", 1, 140),
    3: AtomicData("Li", 0.98, 182),
    4: AtomicData("Be", 1.57, 100),
    5: AtomicData("B", 2.04, 100),
    6: AtomicData("C", 2.55, 170),
    7: AtomicData("N", 3.04, 155),
    8: AtomicData("O", 3.44, 152),
    9: AtomicData("F", 3.98, 147),
    10: AtomicData("Ne", 1, 154),
    11: AtomicData("Na", 0.93, 227),
    12: AtomicData("Mg", 1.31, 173),
    13: AtomicData("Al", 1.61, 100),
    14: AtomicData("Si", 1.90, 210),
    15: AtomicData("P", 2.19, 180),
    16: AtomicData("S", 2.58, 180),
    17: AtomicData("Cl", 3.16, 175),
    18: AtomicData("Ar", 1, 188),
    19: AtomicData("K", 0.82, 275),
    20: AtomicData("Ca", 1.00, 200),
    31: AtomicData("Ga", 1.81, 187),
    32: AtomicData("Ge", 2.01, 200),
    33: AtomicData("As", 2.18, 185),
    34: AtomicData("Se", 2.55, 190),
    35: AtomicData("Br", 2.96, 185),
    36: AtomicData("Kr", 1, 202),
    37: AtomicData("Rb", 0.82, 200),
    38: AtomicData("Sr", 0.95, 200),
    49: AtomicData("In", 1.78, 193),
    50: AtomicData("Sn", 1.96, 217),
    51: AtomicData("Sb", 2.05, 200),
    52: AtomicData("Te", 2.1, 206),
    53: AtomicData("I", 2.66, 198),
    54: AtomicData("Xe", 1, 216),
    55: AtomicData("Cs", 0.79, 200),
    56: AtomicData("Ba", 0.89, 200),
    81: AtomicData("Tl", 1.62, 196),
    82: AtomicData("Pb", 2.33, 202),
    83: AtomicData("Bi", 2.02, 200),
}
