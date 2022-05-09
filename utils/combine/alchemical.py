import torch
import numpy as np

from equistore import TensorMap, Labels, TensorBlock

from collections import namedtuple
import sklearn.decomposition


class CombineSpecies(torch.nn.Module):
    def __init__(self, species, n_pseudo_species, *, explicit_combining_matrix=None):
        super().__init__()
        coupling = _species_coupling_matrix(species)
        pca = sklearn.decomposition.PCA(n_components=n_pseudo_species)

        self.species_remapping = {species: i for i, species in enumerate(species)}

        if explicit_combining_matrix is None:
            self.combining_matrix = torch.nn.Parameter(
                torch.tensor(pca.fit_transform(coupling))
                .contiguous()
                .to(dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineSpecies(
            list(self.species_remapping.keys()),
            self.combining_matrix.shape[1],
            explicit_combining_matrix=self.combining_matrix.clone().detach()
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("neighbor_species", "n")

        n_species, n_pseudo_species = self.combining_matrix.shape

        blocks = []
        for _, block in spherical_expansion:
            radial = np.unique(block.properties["n"])
            n_radial = len(radial)
            properties = Labels(
                names=["neighbor_species", "n"],
                values=np.array(
                    [[-s, n] for s in range(n_pseudo_species) for n in radial],
                    dtype=np.int32,
                ),
            )

            n_samples, n_components, _ = block.values.shape
            data = block.values.reshape(n_samples, n_components, n_species, n_radial)

            data = data.swapaxes(-1, -2)
            data = data @ self.combining_matrix

            data = data.swapaxes(-1, -2)
            data = data.reshape(n_samples, n_components, n_radial * n_pseudo_species)

            new_block = TensorBlock(
                values=data,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")

                n_grad_samples, n_cartesian, n_spherical, _ = gradient.data.shape

                data = gradient.data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_species,
                    n_radial,
                )

                data = data.swapaxes(-1, -2)
                data = data @ self.combining_matrix

                data = data.swapaxes(-1, -2)
                data = data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_radial * n_pseudo_species,
                )

                new_block.add_gradient(
                    "positions",
                    data,
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)


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
            a = delta_epsilon**2 / (2 * SIGMA_EPSILON**2)

            delta_radius = constants_i.radius - constants_j.radius
            b = delta_radius**2 / (2 * SIGMA_RADIUS**2)

            K[i, j] = np.exp(-(a + b))

    return K


AtomicData = namedtuple("AtomicData", ["symbol", "electronegativity", "radius"])

ATOMIC_DATA_PER_SPECIES = {
    1: AtomicData("H", 2.20, 31.0),
    2: AtomicData("He", 1.0, 28.0),
    3: AtomicData("Li", 0.98, 128.0),
    4: AtomicData("Be", 1.57, 96.0),
    5: AtomicData("B", 2.04, 84.0),
    6: AtomicData("C", 2.55, 76.0),
    7: AtomicData("N", 3.04, 71.0),
    8: AtomicData("O", 3.44, 66.0),
    9: AtomicData("F", 3.98, 57.0),
    10: AtomicData("Ne", 1.0, 58.0),
    11: AtomicData("Na", 0.93, 166.0),
    12: AtomicData("Mg", 1.31, 141.0),
    13: AtomicData("Al", 1.61, 121.0),
    14: AtomicData("Si", 1.90, 111.0),
    15: AtomicData("P", 2.19, 107.0),
    16: AtomicData("S", 2.58, 105.0),
    17: AtomicData("Cl", 3.16, 102.0),
    18: AtomicData("Ar", 1.0, 106.0),
    19: AtomicData("K", 0.82, 203.0),
    20: AtomicData("Ca", 1.0, 176.0),
    21: AtomicData("Sc", 1.36, 170.0),
    22: AtomicData("Ti", 1.54, 160.0),
    23: AtomicData("V", 1.63, 153.0),
    24: AtomicData("Cr", 1.66, 139.0),
    25: AtomicData("Mn", 1.55, 139.0),
    26: AtomicData("Fe", 1.83, 132.0),
    27: AtomicData("Co", 1.88, 126.0),
    28: AtomicData("Ni", 1.91, 124.0),
    29: AtomicData("Cu", 1.90, 132.0),
    30: AtomicData("Zn", 1.65, 122.0),
    31: AtomicData("Ga", 1.81, 122.0),
    32: AtomicData("Ge", 2.01, 120.0),
    33: AtomicData("As", 2.18, 119.0),
    34: AtomicData("Se", 2.55, 120.0),
    35: AtomicData("Br", 2.96, 120.0),
    36: AtomicData("Kr", 3.0, 116.0),
    37: AtomicData("Rb", 0.82, 220.0),
    38: AtomicData("Sr", 0.95, 195.0),
    39: AtomicData("Y", 1.22, 190.0),
    40: AtomicData("Zr", 1.33, 175.0),
    41: AtomicData("Nb", 1.6, 164.0),
    42: AtomicData("Mo", 2.16, 154.0),
    43: AtomicData("Tc", 1.9, 147.0),
    44: AtomicData("Ru", 2.2, 146.0),
    45: AtomicData("Rh", 2.28, 142.0),
    46: AtomicData("Pd", 2.20, 139.0),
    47: AtomicData("Ag", 1.93, 145.0),
    48: AtomicData("Cd", 1.69, 144.0),
    49: AtomicData("In", 1.78, 142.0),
    50: AtomicData("Sn", 1.96, 139.0),
    51: AtomicData("Sb", 2.05, 139.0),
    52: AtomicData("Te", 2.1, 138.0),
    53: AtomicData("I", 2.66, 139.0),
    54: AtomicData("Xe", 2.6, 140.0),
    55: AtomicData("Cs", 0.79, 244.0),
    56: AtomicData("Ba", 0.89, 215.0),
    57: AtomicData("La", 1.10, 207.0),
    58: AtomicData("Ce", 1.12, 204.0),
    59: AtomicData("Pr", 1.13, 203.0),
    60: AtomicData("Nd", 1.14, 201.0),
    61: AtomicData("Pm", 1.0, 199.0),
    62: AtomicData("Sm", 1.17, 198.0),
    63: AtomicData("Eu", 1.0, 198.0),
    64: AtomicData("Gd", 1.20, 196.0),
    65: AtomicData("Tb", 1.0, 194.0),
    66: AtomicData("Dy", 1.22, 192.0),
    67: AtomicData("Ho", 1.23, 192.0),
    68: AtomicData("Er", 1.24, 189.0),
    69: AtomicData("Tm", 1.25, 190.0),
    70: AtomicData("Yb", 1.0, 187.0),
    71: AtomicData("Lu", 1.27, 187.0),
    72: AtomicData("Hf", 1.3, 175.0),
    73: AtomicData("Ta", 1.5, 170.0),
    74: AtomicData("W", 2.36, 162.0),
    75: AtomicData("Re", 1.9, 151.0),
    76: AtomicData("Os", 2.2, 144.0),
    77: AtomicData("Ir", 2.20, 141.0),
    78: AtomicData("Pt", 2.28, 136.0),
    79: AtomicData("Au", 2.54, 136.0),
    80: AtomicData("Hg", 2.0, 132.0),
    81: AtomicData("Tl", 1.62, 145.0),
    82: AtomicData("Pb", 2.33, 146.0),
    83: AtomicData("Bi", 2.02, 148.0),
    84: AtomicData("Po", 2.0, 140.0),
    85: AtomicData("At", 2.2, 150.0),
}
