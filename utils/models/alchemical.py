import torch

from ..soap import PowerSpectrum
from ..alchemical import AlchemicalCombine

from ..gap import FullGap, FullLinearGap, SparseGap, RidgeRegression


class BaseMixedSpeciesGapModel_(torch.nn.Module):
    def __init__(
        self,
        species,
        n_pseudo_species,
        uses_support_points,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
    ):
        super().__init__()
        self.power_spectrum = PowerSpectrum()
        self.alchemical = AlchemicalCombine(species, n_pseudo_species)

        self.uses_support_points = uses_support_points
        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

        self.detach_support_points = detach_support_points

        self.model = None

    def _fit_model(self, power_spectrum, all_species, structures_slices, energies):
        raise Exception("this should be implemented in the child class")

    def update_support_points(
        self,
        spherical_expansion,
        all_species,
        structures_slices,
        select_again=False,
    ):
        combined = self.alchemical(spherical_expansion)
        power_spectrum = self.power_spectrum(combined)
        self.model.update_support_points(
            power_spectrum,
            all_species,
            structures_slices,
            select_again,
        )

    def fit(self, spherical_expansion, all_species, structures_slices, energies):
        if self.model is not None and self.optimizable_weights:
            raise Exception("You should only call fit once with optimizable weights")

        combined = self.alchemical(spherical_expansion)
        power_spectrum = self.power_spectrum(combined)
        self.model = self._fit_model(
            power_spectrum, all_species, structures_slices, energies
        )

    def forward(self, spherical_expansion, all_species, structures_slices):
        combined = self.alchemical(spherical_expansion)
        ps = self.power_spectrum(combined)
        return self.model(ps, all_species, structures_slices)

    def parameters(self):
        if self.model is None and self.optimizable_weights:
            raise Exception("please call fit before calling parameters")

        return super().parameters()


class MixedSpeciesLinearModel(BaseMixedSpeciesGapModel_):
    def __init__(
        self,
        species,
        n_pseudo_species,
        lambdas,
        optimizable_weights=False,
        random_initial_weights=False,
    ):
        super().__init__(
            species,
            n_pseudo_species,
            uses_support_points=False,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
            detach_support_points=True,
        )

        self.lambdas = lambdas

    def _fit_model(self, power_spectrum, all_species, structures_slices, energies):
        return RidgeRegression(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            lambdas=self.lambdas,
            optimizable_weights=self.optimizable_weights,
            random_initial_weights=self.random_initial_weights,
        )


class MixedSpeciesFullGapModel(BaseMixedSpeciesGapModel_):
    def __init__(
        self,
        species,
        n_pseudo_species,
        zeta,
        lambdas,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
    ):
        super().__init__(
            species,
            n_pseudo_species,
            uses_support_points=True,
            optimizable_weights=optimizable_weights,
            detach_support_points=detach_support_points,
            random_initial_weights=random_initial_weights,
        )

        self.zeta = zeta
        self.lambdas = lambdas

    def _fit_model(self, power_spectrum, all_species, structures_slices, energies):
        return FullGap(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            zeta=self.zeta,
            lambdas=self.lambdas,
            optimizable_weights=self.optimizable_weights,
            detach_support_points=self.detach_support_points,
            random_initial_weights=self.random_initial_weights,
        )


class MixedSpeciesFullLinearGapModel(BaseMixedSpeciesGapModel_):
    def __init__(
        self,
        species,
        n_pseudo_species,
        lambdas,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
    ):
        super().__init__(
            species,
            n_pseudo_species,
            uses_support_points=True,
            optimizable_weights=optimizable_weights,
            detach_support_points=detach_support_points,
            random_initial_weights=random_initial_weights,
        )

        self.lambdas = lambdas

    def _fit_model(self, power_spectrum, all_species, structures_slices, energies):
        return FullLinearGap(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            lambdas=self.lambdas,
            optimizable_weights=self.optimizable_weights,
            detach_support_points=self.detach_support_points,
            random_initial_weights=self.random_initial_weights,
        )


class MixedSpeciesSparseGapModel(BaseMixedSpeciesGapModel_):
    def __init__(
        self,
        species,
        n_pseudo_species,
        n_support,
        zeta,
        lambdas,
        optimizable_weights,
        random_initial_weights,
        detach_support_points,
        jitter=1e-12,
    ):
        super().__init__(
            species,
            n_pseudo_species,
            uses_support_points=True,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
            detach_support_points=detach_support_points,
        )

        self.zeta = zeta
        self.lambdas = lambdas
        self.n_support = n_support
        self.jitter = jitter

    def _fit_model(self, power_spectrum, all_species, structures_slices, energies):
        return SparseGap(
            power_spectrum=power_spectrum,
            structures_slices=structures_slices,
            energies=energies,
            n_support=self.n_support,
            zeta=self.zeta,
            lambdas=self.lambdas,
            jitter=self.jitter,
            optimizable_weights=self.optimizable_weights,
            random_initial_weights=self.random_initial_weights,
            detach_support_points=self.detach_support_points,
        )
