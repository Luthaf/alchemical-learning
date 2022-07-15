import torch

from utils.linear import LinearModel
from utils.operations import SumStructures, remove_gradient
from utils.soap import PowerSpectrum

# only profile when required
try:
    profile
except NameError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


MB_DEFAULT_OPTIONS = {}


class CombinedPowerSpectrum(torch.nn.Module):
    def __init__(self, combiner):
        super().__init__()

        self.combiner = combiner
        self.power_spectrum = PowerSpectrum()

    @profile
    def forward(self, spherical_expansion):
        combined = self.combiner(spherical_expansion)

        ps = self.power_spectrum(combined)
        return ps


class MultiBodyOrderModel(torch.nn.Module):
    def __init__(
        self,
        power_spectrum,
        composition_regularizer,
        radial_spectrum_regularizer,
        power_spectrum_regularizer,
        optimizable_weights,
        random_initial_weights,
        # list of atomic types to explode the power spectrum. None to use same
        # model for all centers (default)
        ps_center_types=None,
    ):
        super().__init__()

        # optimizable_weights = False is not very well tested ...
        assert optimizable_weights

        if composition_regularizer is None:
            self.composition_model = None
        else:
            self.composition_model = LinearModel(
                regularizer=composition_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        if radial_spectrum_regularizer is None:
            self.radial_spectrum_model = None
        else:
            self.radial_spectrum_model = LinearModel(
                regularizer=radial_spectrum_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        if power_spectrum_regularizer is None:
            self.power_spectrum_model = None
        else:
            self.sum_structure = SumStructures(explode_centers=ps_center_types)
            self.power_spectrum = power_spectrum
            self.power_spectrum_model = LinearModel(
                regularizer=power_spectrum_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

    @profile
    def forward(
        self, composition, radial_spectrum, spherical_expansion, forward_forces=False
    ):
        if not forward_forces:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)

        energies, forces = None, None

        if self.composition_model is not None:
            energies_cmp, _ = self.composition_model(composition)
            energies = energies_cmp
            forces = None

        if self.radial_spectrum_model is not None:
            radial_spectrum_per_structure = (
                radial_spectrum  # self.sum_structure(radial_spectrum)
            )
            energies_rs, forces_rs = self.radial_spectrum_model(
                radial_spectrum_per_structure, with_forces=forward_forces
            )

            if energies is None:
                energies = energies_rs
            else:
                energies += energies_rs
            if forces_rs is not None:
                if forces is None:
                    forces = forces_rs
                else:
                    forces += forces_rs

        if self.power_spectrum_model is not None:
            power_spectrum = self.power_spectrum(spherical_expansion)
            power_spectrum_per_structure = self.sum_structure(power_spectrum)

            energies_ps, forces_ps = self.power_spectrum_model(
                power_spectrum_per_structure, with_forces=forward_forces
            )
            if energies is None:
                energies = energies_ps
            else:
                energies += energies_ps
            if forces_ps is not None:
                if forces is None:
                    forces = forces_ps
                else:
                    forces += forces_ps

        return energies, forces

    def initialize_model_weights(
        self,
        composition,
        radial_spectrum,
        spherical_expansion,
        energies,
        forces=None,
        seed=None,
    ):
        if forces is None:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)

        if self.composition_model is not None:
            self.composition_model.initialize_model_weights(
                composition, energies, forces, seed
            )

        if self.radial_spectrum_model is not None:
            radial_spectrum_per_structure = radial_spectrum
            self.radial_spectrum_model.initialize_model_weights(
                radial_spectrum_per_structure, energies, forces, seed
            )

        if self.power_spectrum_model is not None:
            power_spectrum = self.power_spectrum(spherical_expansion)
            power_spectrum_per_structure = self.sum_structure(power_spectrum)
            self.power_spectrum_model.initialize_model_weights(
                power_spectrum_per_structure, energies, forces, seed
            )
