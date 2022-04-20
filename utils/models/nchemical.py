import torch

from ..soap import PowerSpectrum
from ..alchemical import NChemicalCombine

from ..gap import RidgeRegression


class BaseNChemicalGapModel_(torch.nn.Module):
    def __init__(
        self,
        spherical_expansion,
        n_contracted,
        uses_support_points,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
    ):
        super().__init__()
        self.power_spectrum = PowerSpectrum()
        self.nchemical = NChemicalCombine(spherical_expansion, n_contracted)

        self.uses_support_points = uses_support_points
        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

        self.detach_support_points = detach_support_points

        self.model = None

    def _fit_model(
        self,
        power_spectrum,
        all_species,
        structures_slices,
        energies,
        forces,
    ):
        raise Exception("this should be implemented in the child class")

    def update_support_points(
        self, spherical_expansion, all_species, select_again=False
    ):
        combined = self.nchemical(spherical_expansion)
        power_spectrum = self.power_spectrum(combined)
        self.model.update_support_points(power_spectrum, all_species, select_again)

    def fit(
        self,
        spherical_expansion,
        all_species,
        structures_slices,
        energies,
        forces=None,
    ):
        if self.model is not None and self.optimizable_weights:
            raise Exception("You should only call fit once with optimizable weights")

        combined = self.nchemical(spherical_expansion)
        power_spectrum = self.power_spectrum(combined)
        self.model = self._fit_model(
            power_spectrum, all_species, structures_slices, energies, forces
        )

    def forward(self, spherical_expansion, all_species, structures_slices):
        combined = self.nchemical(spherical_expansion)
        ps = self.power_spectrum(combined)
        return self.model(ps, all_species, structures_slices)

    def parameters(self):
        if self.model is None and self.optimizable_weights:
            raise Exception("please call fit before calling parameters")

        return super().parameters()


class NChemicalLinearModel(BaseNChemicalGapModel_):
    def __init__(
        self,
        spherical_expansion,
        n_contracted,
        lambdas,
        optimizable_weights=False,
        random_initial_weights=False,
    ):
        super().__init__(
            spherical_expansion,
            n_contracted,
            uses_support_points=False,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
            detach_support_points=True,
        )

        self.lambdas = lambdas

    def _fit_model(
        self,
        power_spectrum,
        all_species,
        structures_slices,
        energies,
        forces,
    ):
        return RidgeRegression(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            forces=forces,
            lambdas=self.lambdas,
            optimizable_weights=self.optimizable_weights,
            random_initial_weights=self.random_initial_weights,
        )
