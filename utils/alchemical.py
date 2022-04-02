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

class RadialChemicalCombine(torch.nn.Module):
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
1: AtomicData("H", 2.20, 31.),
2: AtomicData("He", 1., 28.),
3: AtomicData("Li", 0.98, 128.),
4: AtomicData("Be", 1.57, 96.),
5: AtomicData("B", 2.04, 84.),
6: AtomicData("C", 2.55, 76.),
7: AtomicData("N", 3.04, 71.),
8: AtomicData("O", 3.44, 66.),
9: AtomicData("F", 3.98, 57.),
10: AtomicData("Ne", 1., 58.),
11: AtomicData("Na", 0.93, 166.),
12: AtomicData("Mg", 1.31, 141.),
13: AtomicData("Al", 1.61, 121.),
14: AtomicData("Si", 1.90, 111.),
15: AtomicData("P", 2.19, 107.),
16: AtomicData("S", 2.58, 105.),
17: AtomicData("Cl", 3.16, 102.),
18: AtomicData("Ar", 1., 106.),
19: AtomicData("K", 0.82, 203.),
20: AtomicData("Ca", 1.0, 176.),
21: AtomicData("Sc", 1.36, 170.),
22: AtomicData("Ti", 1.54, 160.),
23: AtomicData("V", 1.63, 153.),
24: AtomicData("Cr", 1.66, 139.),
25: AtomicData("Mn", 1.55, 139.),
26: AtomicData("Fe", 1.83, 132.),
27: AtomicData("Co", 1.88, 126.),
28: AtomicData("Ni", 1.91, 124.),
29: AtomicData("Cu", 1.90, 132.),
30: AtomicData("Zn", 1.65, 122.),
31: AtomicData("Ga", 1.81, 122.),
32: AtomicData("Ge", 2.01, 120.),
33: AtomicData("As", 2.18, 119.),
34: AtomicData("Se", 2.55, 120.),
35: AtomicData("Br", 2.96, 120.),
36: AtomicData("Kr", 3.0, 116.),
37: AtomicData("Rb", 0.82, 220.),
38: AtomicData("Sr", 0.95, 195.),
39: AtomicData("Y", 1.22, 190.),
40: AtomicData("Zr", 1.33, 175.),
41: AtomicData("Nb", 1.6, 164.),
42: AtomicData("Mo", 2.16, 154.),
43: AtomicData("Tc", 1.9, 147.),
44: AtomicData("Ru", 2.2, 146.),
45: AtomicData("Rh", 2.28, 142.),
46: AtomicData("Pd", 2.20, 139.),
47: AtomicData("Ag", 1.93, 145.),
48: AtomicData("Cd", 1.69, 144.),
49: AtomicData("In", 1.78, 142.),
50: AtomicData("Sn", 1.96, 139.),
51: AtomicData("Sb", 2.05, 139.),
52: AtomicData("Te", 2.1, 138.),
53: AtomicData("I", 2.66, 139.),
54: AtomicData("Xe", 2.6, 140.),
55: AtomicData("Cs", 0.79, 244.),
56: AtomicData("Ba", 0.89, 215.),
57: AtomicData("La", 1.10, 207.),
58: AtomicData("Ce", 1.12, 204.),
59: AtomicData("Pr", 1.13, 203.),
60: AtomicData("Nd", 1.14, 201.),
61: AtomicData("Pm", 1., 199.),
62: AtomicData("Sm", 1.17, 198.),
63: AtomicData("Eu", 1., 198.),
64: AtomicData("Gd", 1.20, 196.),
65: AtomicData("Tb", 1., 194.),
66: AtomicData("Dy", 1.22, 192.),
67: AtomicData("Ho", 1.23, 192.),
68: AtomicData("Er", 1.24, 189.),
69: AtomicData("Tm", 1.25, 190.),
70: AtomicData("Yb", 1., 187.),
71: AtomicData("Lu", 1.27, 187.),
72: AtomicData("Hf", 1.3, 175.),
73: AtomicData("Ta", 1.5, 170.),
74: AtomicData("W", 2.36, 162.),
75: AtomicData("Re", 1.9, 151.),
76: AtomicData("Os", 2.2, 144.),
77: AtomicData("Ir", 2.20, 141.),
78: AtomicData("Pt", 2.28, 136.),
79: AtomicData("Au", 2.54, 136.),
80: AtomicData("Hg", 2.0, 132.),
81: AtomicData("Tl", 1.62, 145.),
82: AtomicData("Pb", 2.33, 146.),
83: AtomicData("Bi", 2.02, 148.),
84: AtomicData("Po", 2.0, 140.),
85: AtomicData("At", 2.2, 150.),
}
