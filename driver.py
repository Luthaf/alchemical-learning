"""Generic calculator-style interface for MD"""
import json

import ase.io
import torch
from rascaline_torch import as_torch_system

from utils.combine import CombineSpecies
from utils.dataset import AtomisticDataset, create_dataloader
from utils.model import AlchemicalModel

torch.set_default_dtype(torch.float64)


device = "cpu"

all_species = [
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    71,
    72,
    73,
    74,
    77,
    78,
    79,
]


class GenericMDCalculator:

    matrix_indices_in_voigt_notation = (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    )

    def __init__(
        self,
        model_state_path,
        model_parameters_path,
        structure_template=None,
        starting_frame=None,
    ):
        super().__init__()

        if structure_template is not None:
            self.template_filename = structure_template
            self.atoms = ase.io.read(structure_template, 0)
            self.atoms.pbc = [True, True, True]
        elif starting_frame is not None:
            self.atoms = starting_frame
        else:
            raise ValueError(
                "Must specify one of 'structure_template' or 'atomic_numbers'"
            )

        with open(model_parameters_path) as fd:
            self._params = json.load(fd)

        self.hypers = {}
        if "hypers_ps" in self._params:
            hypers_ps = self._params["hypers_ps"]
            if "radial_per_angular" in hypers_ps:
                hypers_ps["radial_per_angular"] = {
                    int(l): n for l, n in hypers_ps["radial_per_angular"].items()
                }

            self.hypers["spherical_expansion"] = hypers_ps

        if "hypers_rs" in self._params:
            self.hypers["radial_spectrum"] = self._params["hypers_rs"]

        self._dataset = AtomisticDataset(
            [self.atoms],
            all_species,
            self.hypers,
            torch.tensor([[0.0]]),
            do_gradients=False,
        )

        dataloader = create_dataloader(
            self._dataset,
            batch_size=1,
            shuffle=False,
            device=device,
        )

        combiner = CombineSpecies(
            species=all_species,
            n_pseudo_species=self._params["n_pseudo_species"],
        )

        self.model = AlchemicalModel(
            combiner=combiner,
            composition_regularizer=self._params.get("composition_regularizer"),
            radial_spectrum_regularizer=self._params.get("radial_spectrum_regularizer"),
            power_spectrum_regularizer=self._params.get("power_spectrum_regularizer"),
            nn_layer_size=self._params.get("nn_layer_size", 0),
            optimizable_weights=True,
            random_initial_weights=True,
        )

        self.model.to(device=device, dtype=torch.get_default_dtype())

        # initialize the model
        with torch.no_grad():
            for (
                composition,
                radial_spectrum,
                spherical_expansions,
                energies,
                _,
            ) in dataloader:
                self.model.initialize_model_weights(
                    composition, radial_spectrum, spherical_expansions, energies
                )

        self.model.load_state_dict(torch.load(model_state_path, map_location=device))
        self.model.eval()

    def calculate(self, positions, cell_matrix):
        """Calculate energies and forces from position/cell update

        positions   Atomic positions (Nx3 matrix)
        cell_matrix Unit cell (in ASE format, cell vectors as rows)
                    (set to zero for non-periodic simulations)

        The units of positions and cell are determined by the model JSON
        file; for now, only Å is supported.  Energies, forces, and
        stresses are returned in the same units (eV and Å supported).

        Returns a tuple of energy, forces, and stress - forces are
        returned as an Nx3 array and stresses are returned as a 3x3 array

        Stress convention: The stresses have units eV/Å^3
        (volume-normalized) and are defined as the gradients of the
        energy with respect to the cell parameters.
        """

        # Quick consistency checks
        if positions.shape != (len(self.atoms), 3):
            raise ValueError(
                "Improper shape of positions (is the number of atoms consistent?)"
            )
        if cell_matrix.shape != (3, 3):
            raise ValueError("Improper shape of cell info (expected 3x3 matrix)")

        # Update ASE Atoms object
        self.atoms.set_cell(cell_matrix)
        self.atoms.set_positions(positions)

        # Compute representations and evaluate model
        # TODO: handle different devices
        composition = self._dataset.compute_composition(self.atoms, 0)
        torch_system = as_torch_system(self.atoms)
        torch_system.positions.requires_grad_(True)
        torch_system.cell.requires_grad_(True)

        radial_spectrum = self._dataset.compute_radial_spectrum(torch_system, 0)
        spherical_expansion = self._dataset.compute_spherical_expansion(torch_system, 0)

        energy_torch, _ = self.model(
            composition,
            radial_spectrum,
            spherical_expansion,
            forward_forces=False,
        )

        # backward propagation for forces and stress
        energy_torch.backward()

        forces_torch = -torch_system.positions.grad
        cell_grad = torch_system.cell.grad

        energy = energy_torch.detach().numpy()
        forces = forces_torch.numpy()

        # Computes the virial as -dU/de (e is the strain)
        virial = -cell_grad.numpy().T @ cell_matrix
        # Symmetrize the virial (should already be almost symmetric, this is a
        # good check)
        virial = 0.5 * (virial + virial.T)

        return energy, forces, virial
