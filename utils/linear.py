import numpy as np
import torch


class LinearModel(torch.nn.Module):
    def __init__(
        self,
        regularizer,
        optimizable_weights=False,
        random_initial_weights=False,
    ):
        super().__init__()

        self.regularizer = regularizer

        self.weights = None

        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

    def initialize_model_weights(self, descriptor, energies, forces=None):

        if self.random_initial_weights:
            X = descriptor.block().values
            weights = torch.rand((X.shape[1], 1), device=X.device)
        else:
            weights = self._fit_linear_model(descriptor, energies, forces)

        if self.optimizable_weights:
            self.weights = torch.nn.Parameter(weights.detach())
        else:
            self.weights = weights

    def _fit_linear_model(self, descriptor, energies, forces):
        block = descriptor.block()
        assert len(block.components) == 0

        X = block.values

        Y = energies.reshape(-1, 1)

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

        weights = X.T @ torch.linalg.solve(X_XT, Y)

        return weights

    def forward(self, descriptor, with_forces=False):
        if self.weights is None:
            raise Exception("call initialize_weights first")

        block = descriptor.block()
        assert len(block.components) == 0

        X = block.values

        energies = X @ self.weights

        if with_forces:
            gradient = block.gradient("positions")
            X_grad = gradient.data.reshape(-1, 3, self.weights.shape[0])

            forces = -X_grad @ self.weights
            forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces
