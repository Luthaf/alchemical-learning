import numpy as np
import torch

from .operations import structure_sum, normalize


class LinearModel(torch.nn.Module):
    def __init__(
        self,
        normalize,
        regularizer,
        optimizable_weights=False,
        random_initial_weights=False,
    ):
        super().__init__()

        self.regularizer = regularizer
        self.normalize = normalize

        self.weights = None
        self.baseline = 0.0

        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

    def initialize_parameters(self, power_spectrum, energies, forces=None):
        # TODO: do we want a baseline?
        # self.baseline = energies.mean()

        if self.random_initial_weights:
            ps = power_spectrum.block().values
            weights = torch.rand((ps.shape[1], 1), device=ps.device)
        else:
            weights = self._fit_linear_model(power_spectrum, energies, forces)

        if self.optimizable_weights:
            self.weights = torch.nn.Parameter(weights.detach())
        else:
            self.weights = weights

    def _fit_linear_model(self, power_spectrum, energies, forces):
        ps_per_structure = structure_sum(power_spectrum)

        if self.normalize:
            ps_per_structure = normalize(ps_per_structure)

        block = ps_per_structure.block()
        assert len(block.components) == 0

        X = block.values

        Y = energies.reshape(-1, 1) - self.baseline

        delta = energies.std()
        structures = np.unique(block.samples["structure"])
        n_atoms_per_structure = []
        for structure in structures:
            n_atoms = np.sum(block.samples["structure"] == structure)
            n_atoms_per_structure.append(float(n_atoms))

        energy_regularizer = (
            torch.sqrt(torch.tensor(n_atoms_per_structure, device=energies.device))
            * self.regularizer[0]
            / delta
        )

        if forces is not None:
            gradient = block.gradient("positions")
            X_grad = gradient.data.reshape(3 * len(gradient.samples), X.shape[1])

            energy_grad = -forces.reshape(X_grad.shape[0], 1)

            # TODO: this assume the atoms are in the same order in X_grad &
            # forces
            Y = torch.vstack([Y, energy_grad])
            X = torch.vstack([X, X_grad])

        # solve weights as `w = X.T (X X.T + λ I)^{-1} Y` instead of the usual
        # `w = (X.T X + λ I)^{-1} X.T Y` since that allow us to regularize
        # energy & forces separately.
        #
        # cf https://stats.stackexchange.com/a/134068
        X_XT = X @ X.T

        property_idx = np.arange(len(energies))
        X_XT[property_idx, property_idx] += energy_regularizer

        if forces is not None:
            forces_regularizer = self.regularizer[1] / delta
            grad_idx = np.arange(len(energies), stop=X_XT.shape[0])
            X_XT[grad_idx, grad_idx] += forces_regularizer

        weights = X.T @ torch.linalg.inv(X_XT) @ Y

        return weights

    def forward(self, power_spectrum, with_forces=False):
        if self.weights is None:
            raise Exception("call initialize_weights first")

        ps_per_structure = structure_sum(power_spectrum)

        if self.normalize:
            ps_per_structure = normalize(ps_per_structure)

        block = ps_per_structure.block()
        assert len(block.components) == 0

        X = block.values

        energies = X @ self.weights + self.baseline

        if with_forces:
            raise ValueError("not checked yet")
            # gradient = block.gradient("positions")
            # X_grad = gradient.data.reshape(-1, 3, self.weights.shape[0])

            # forces = -X_grad @ self.weights
            # forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces
