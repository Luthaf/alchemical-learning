import torch

from ..soap import PowerSpectrum
from ..gap import FullGap, SparseGap, SparseGapPerSpecies, RidgeRegression


class BaseGapModel_(torch.nn.Module):
    def __init__(
        self,
        uses_support_points,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
    ):
        super().__init__()
        self.power_spectrum = PowerSpectrum()

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

        power_spectrum = self.power_spectrum(spherical_expansion)
        self.model = self._fit_model(
            power_spectrum, all_species, structures_slices, energies, forces
        )

    def forward(self, spherical_expansion, all_species, structures_slices):
        ps = self.power_spectrum(spherical_expansion)
        return self.model(ps, all_species, structures_slices)

    def parameters(self):
        if self.model is None and self.optimizable_weights:
            raise Exception("please call fit before calling parameters")

        return super().parameters()


class LinearModel(BaseGapModel_):
    def __init__(
        self,
        lambdas,
        optimizable_weights=False,
        random_initial_weights=False,
    ):
        super().__init__(
            uses_support_points=False,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
            detach_support_points=False,
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
            lambdas=self.lambdas,
            optimizable_weights=self.optimizable_weights,
            random_initial_weights=self.random_initial_weights,
        )


class FullGapModel(BaseGapModel_):
    def __init__(
        self,
        zeta,
        lambdas,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
    ):
        super().__init__(
            uses_support_points=True,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
            detach_support_points=detach_support_points,
        )

        self.zeta = zeta
        self.lambdas = lambdas

    def _fit_model(
        self,
        power_spectrum,
        all_species,
        structures_slices,
        energies,
        forces,
    ):
        return FullGap(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            forces=forces,
            zeta=self.zeta,
            lambdas=self.lambdas,
            optimizable_weights=self.optimizable_weights,
            random_initial_weights=self.random_initial_weights,
            detach_support_points=self.detach_support_points,
        )


class SparseGapModel(BaseGapModel_):
    def __init__(
        self,
        n_support,
        zeta,
        lambdas,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
        jitter=1e-12,
    ):
        super().__init__(
            uses_support_points=True,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
            detach_support_points=detach_support_points,
        )

        self.power_spectrum = PowerSpectrum()

        self.n_support = n_support
        self.zeta = zeta
        self.lambdas = lambdas
        self.jitter = jitter
        self.optimizable_weights = optimizable_weights

        self.model = None

    def _fit_model(
        self,
        power_spectrum,
        all_species,
        structures_slices,
        energies,
        forces,
    ):
        return SparseGap(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            forces=forces,
            n_support=self.n_support,
            zeta=self.zeta,
            lambdas=self.lambdas,
            jitter=self.jitter,
            optimizable_weights=self.optimizable_weights,
            random_initial_weights=self.random_initial_weights,
            detach_support_points=self.detach_support_points,
        )


class PerSpeciesSparseGapModel(BaseGapModel_):
    def __init__(
        self, n_support, zeta, lambdas, jitter=1e-12, optimizable_weights=False
    ):
        super().__init__(
            uses_support_points=True,
            optimizable_weights=optimizable_weights,
        )

        self.power_spectrum = PowerSpectrum()

        self.n_support = n_support
        self.zeta = zeta
        self.lambdas = lambdas
        self.jitter = jitter
        self.optimizable_weights = optimizable_weights

        self.model = None

    def _fit_model(
        self,
        power_spectrum,
        all_species,
        structures_slices,
        energies,
        forces,
    ):
        return SparseGapPerSpecies(
            power_spectrum=power_spectrum,
            all_species=all_species,
            structures_slices=structures_slices,
            energies=energies,
            forces=forces,
            n_support=self.n_support,
            zeta=self.zeta,
            lambdas=self.lambdas,
            jitter=self.jitter,
            optimizable_weights=self.optimizable_weights,
        )
