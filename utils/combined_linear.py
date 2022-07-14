import numpy as np
import torch
from .operations import SumStructures, remove_gradient
from .soap import PowerSpectrum
from .linear import LinearModel

class CombinedLinearModel(torch.nn.Module):
    def __init__(self, 
        combiner,
        regularizer,
        optimizable_weights,
        random_initial_weights,
    ):
        super().__init__()

        self.sum_structure = SumStructures()
        self.combiner = combiner
        self.power_spectrum = PowerSpectrum()
        self.model = LinearModel(
            regularizer=regularizer,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
        )

        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

    def forward(self, spherical_expansion, forward_forces=False):
        if not forward_forces:
            # remove gradients from the spherical expansion if we don't need it
            spherical_expansion = remove_gradient(spherical_expansion)

        combined = self.combiner(spherical_expansion)
        power_spectrum = self.power_spectrum(combined)        
        power_spectrum_per_structure = self.sum_structure(power_spectrum)
        energies, forces = self.model(power_spectrum_per_structure, with_forces=forward_forces)
        return energies, forces

    def initialize_model_weights(self, spherical_expansion, energies, forces=None):
        if forces is None:
            # remove gradients from the spherical expansion if we don't need it
            spherical_expansion = remove_gradient(spherical_expansion)

        combined = self.combiner(spherical_expansion)
        power_spectrum = self.power_spectrum(combined)
        
        power_spectrum_per_structure = self.sum_structure(power_spectrum)
        self.model.initialize_model_weights(power_spectrum_per_structure, energies, forces)
        