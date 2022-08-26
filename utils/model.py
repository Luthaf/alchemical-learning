import torch

from utils.linear import LinearModel
from utils.nonlinear import NNModel
from utils.operations import SumStructures, remove_gradient
from utils.soap import PowerSpectrum

# only profile when required
try:
    profile
except NameError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


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


class AlchemicalModel(torch.nn.Module):
    def __init__(
        self,
        # combiner for the spherical expansion (mainly tested with alchemical
        # combination)
        combiner,
        # Regularizer for the 1-body/composition model. Set to None to remove
        # the 1-body model from the fit
        composition_regularizer,
        # Regularizer for the 2-body/radial spectrum model. Set to None to remove
        # the 2-body model from the fit
        radial_spectrum_regularizer,
        # Regularizer for the 3-body/power spectrum model. Set to None to remove
        # the 3-body model from the fit
        power_spectrum_regularizer,
        # Number of layers for the neural network applied on top of power spectrum
        # Set to None to use a pure linear model
        nn_layer_size=None,
        # list of atomic types to explode the power spectrum. None to use same
        # model for all centers (default)
        ps_center_types=None,
    ):
        super().__init__()

        if composition_regularizer is None:
            self.composition_model = None
        else:
            self.composition_model = LinearModel(
                regularizer=composition_regularizer,
                optimizable_weights=True,
                random_initial_weights=True,
            )

        if radial_spectrum_regularizer is None:
            self.radial_spectrum_model = None
        else:
            self.radial_spectrum_model = LinearModel(
                regularizer=radial_spectrum_regularizer,
                optimizable_weights=True,
                random_initial_weights=True,
            )

        if power_spectrum_regularizer is None:
            self.power_spectrum_model = None
        else:
            self.sum_structure = SumStructures(explode_centers=ps_center_types)
            self.power_spectrum = CombinedPowerSpectrum(combiner)
            self.power_spectrum_model = LinearModel(
                regularizer=power_spectrum_regularizer,
                optimizable_weights=True,
                random_initial_weights=True,
            )
            if nn_layer_size == 0:
                self.nn_model = None
            else:
                self.nn_model = NNModel(nn_layer_size)

        # self.latest_energies = {}
        # self.latest_forces = {}

    @profile
    def forward(
        self,
        composition,
        radial_spectrum,
        spherical_expansion,
        forward_forces=False,
    ):
        if not forward_forces:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)

        energies = torch.zeros(
            (len(composition.block().samples), 1),
            device=composition.block().values.device,
        )
        forces = None

        if self.composition_model is not None:
            energies_cmp, _ = self.composition_model(composition)
            # self.latest_energies["composition"] = energies_cmp
            energies += energies_cmp

        if self.radial_spectrum_model is not None:
            # Radial spectrum is already summed per-structure
            radial_spectrum_per_structure = radial_spectrum
            energies_rs, forces_rs = self.radial_spectrum_model(
                radial_spectrum_per_structure, with_forces=forward_forces
            )

            # self.latest_energies["radial"] = energies_rs
            # self.latest_forces["radial"] = forces_rs

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

            # self.latest_energies["power"] = energies_ps
            # self.latest_forces["power"] = forces_ps

            energies += energies_ps

            if forces_ps is not None:
                if forces is None:
                    forces = forces_ps
                else:
                    forces += forces_ps

            if self.nn_model is not None:
                nn_energies, nn_forces = self.nn_model(
                    power_spectrum, with_forces=forward_forces
                )
                energies += nn_energies

                # self.latest_energies["nn"] = nn_energies
                # self.latest_forces["nn"] = nn_forces

                if forces is None:
                    forces = nn_forces
                else:
                    forces += nn_forces

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
            if self.nn_model is not None:
                self.nn_model.initialize_model_weights(power_spectrum, energies, forces)
