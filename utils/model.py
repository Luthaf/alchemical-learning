import torch

from utils.linear import LinearModel
from utils.nonlinear import NNModel, NNModelSPeciesWise
from utils.operations import SumStructures, remove_gradient
from utils.multi_species_mlp import MultiSpeciesMLP_skip
from utils.soap import PowerSpectrum
from time import time

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
        optimizable_weights=True,
        random_initial_weights=True,
    ):
        super().__init__()

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
            self.power_spectrum = CombinedPowerSpectrum(combiner)
            self.power_spectrum_model = LinearModel(
                regularizer=power_spectrum_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

            if nn_layer_size == 0:
                self.nn_model = None
            else:
                if not optimizable_weights:
                    raise Exception(
                        "can not use the NN model with optimizable_weights=False"
                    )
                self.nn_model = NNModel(nn_layer_size)

        self._neval = 0        
        self._timings = dict(fw_comp = 0.0, fw_pair = 0.0, fw_ps = 0.0, fw_nn = 0.0)

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
            self._timings["fw_comp"] -= time()
            energies_cmp, _ = self.composition_model(composition)
            energies += energies_cmp
            self._timings["fw_comp"] += time()            

        if self.radial_spectrum_model is not None:
            # Radial spectrum is already summed per-structure
            self._timings["fw_pair"] -= time()
            radial_spectrum_per_structure = radial_spectrum
            energies_rs, forces_rs = self.radial_spectrum_model(
                radial_spectrum_per_structure, with_forces=forward_forces
            )

            energies += energies_rs

            if forces_rs is not None:
                if forces is None:
                    forces = forces_rs
                else:
                    forces += forces_rs
                    
            self._timings["fw_pair"] += time()                    
        
        if self.power_spectrum_model is not None:

            self._timings["fw_ps"] -= time()        
            power_spectrum = self.power_spectrum(spherical_expansion)
            power_spectrum_per_structure = self.sum_structure(power_spectrum)

            energies_ps, forces_ps = self.power_spectrum_model(
                power_spectrum_per_structure, with_forces=forward_forces
            )

            energies += energies_ps

            if forces_ps is not None:
                if forces is None:
                    forces = forces_ps
                else:
                    forces += forces_ps
            self._timings["fw_ps"] += time()        
            
            if self.nn_model is not None:
                self._timings["fw_nn"] -= time()                    
                nn_energies, nn_forces = self.nn_model(
                    power_spectrum, with_forces=forward_forces
                )
                energies += nn_energies

                if forces is None:
                    forces = nn_forces
                else:
                    forces += nn_forces
                self._timings["fw_nn"] += time()                    

        self._neval += 1                
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

            energies -= self.composition_model(composition)[0]

        if self.radial_spectrum_model is not None:
            radial_spectrum_per_structure = radial_spectrum
            self.radial_spectrum_model.initialize_model_weights(
                radial_spectrum_per_structure, energies, forces, seed
            )

            energies -= self.radial_spectrum_model(radial_spectrum_per_structure)[0]

        if self.power_spectrum_model is not None:
            power_spectrum = self.power_spectrum(spherical_expansion)
            power_spectrum_per_structure = self.sum_structure(power_spectrum)
            self.power_spectrum_model.initialize_model_weights(
                power_spectrum_per_structure, energies, forces, seed
            )

            energies -= self.power_spectrum_model(power_spectrum_per_structure)[0]

            if self.nn_model is not None:
                self.nn_model.initialize_model_weights(power_spectrum, energies, forces)


class SoapBpnn(torch.nn.Module):
    def __init__(
        self,
        # combiner for the spherical expansion (mainly tested with alchemical
        # combination)
        combiner,
        # Regularizer for the 1-body/composition model. Set to None to remove
        # the 1-body model from the fit
        composition_regularizer,
        # Number of layers for the neural network applied on top of power spectrum
        # Set to None to use a pure linear model
        nn_layer_size=None,
        # list of atomic types to explode the power spectrum. None to use same
        # model for all centers (default)
        ps_center_types=None,
        optimizable_weights=True,
        random_initial_weights=True,
    ):
        super().__init__()

        if composition_regularizer is None:
            self.composition_model = None
        else:
            self.composition_model = LinearModel(
                regularizer=composition_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        self.power_spectrum = CombinedPowerSpectrum(combiner)
        self.nn_model = NNModelSPeciesWise(nn_layer_size)

        self._neval = 0        
        self._timings = dict(fw_comp = 0.0, fw_pair = 0.0, fw_ps = 0.0, fw_nn = 0.0)

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

        #Model that learns and removes the energy offset, as a function of the structure composition
        if self.composition_model is not None:
            self._timings["fw_comp"] -= time()
            energies_cmp, _ = self.composition_model(composition)
            energies += energies_cmp
            self._timings["fw_comp"] += time()            
            self._timings["fw_pair"] += time()  

        #TODO: Remove this and concatenate RS with SOAP
                    
                              
        
        #TODO: remove the linear part and only apply NN
        self._timings["fw_ps"] -= time()        
        power_spectrum = self.power_spectrum(spherical_expansion)

        
                                 
        
        nn_energies, nn_forces = self.nn_model(
            power_spectrum, with_forces=forward_forces
        )
        
        energies += nn_energies
        forces = nn_forces
        
        self._timings["fw_nn"] += time()                    

        self._neval += 1                
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

            energies -= self.composition_model(composition)[0]

        if self.nn_model is not None:
            power_spectrum = self.power_spectrum(spherical_expansion)
            self.nn_model.initialize_model_weights(power_spectrum, energies, forces)

