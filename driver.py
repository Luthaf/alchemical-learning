"""Generic calculator-style interface for MD"""
import ase.io
import numpy as np
import json

import torch
from utils.dataset import AtomisticDataset, create_dataloader
from utils.soap import PowerSpectrum
from utils.combine import CombineRadial, CombineRadialSpecies, CombineSpecies
from mbmodel import CombinedPowerSpectrum, MultiBodyOrderModel

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
        is_periodic,
        structure_template=None,
        atomic_numbers=None,
    ):
        super().__init__()
        # self.model = torch.load(model_pt)
        # Structure initialization
        self.is_periodic = is_periodic

        if structure_template is not None:
            self.template_filename = structure_template
            self.atoms = ase.io.read(structure_template, 0)
            if (is_periodic is not None) and (
                is_periodic != np.any(self.atoms.get_pbc())
            ):
                raise ValueError(
                    "Structure template PBC flags: "
                    + str(self.atoms.get_pbc())
                    + " incompatible with 'is_periodic' setting"
                )
        elif atomic_numbers is not None:
            self.atoms = ase.Atoms(numbers=atomic_numbers, pbc=is_periodic)
        else:
            raise ValueError(
                "Must specify one of 'structure_template' or 'atomic_numbers'"
            )

        frames = [self.atoms]
        print(len(self.atoms))
        energies = torch.tensor([[0.0]])
        forces = [torch.tensor(np.zeros((len(frames[0]), 3)))]

        self.model_parameters = json.load(
            open(model_parameters_path),
            object_hook=lambda d: {
                int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
            },
        )

        pars_dict = self.model_parameters
        train_normalization = (
            pars_dict["normalization"] if "normalization" in pars_dict else None
        )
        rng_seed = (
            pars_dict["seed"] if "seed" in pars_dict else None
        )  # defaults to random seed
        per_center_ps = (
            pars_dict["per_center_ps"] if "per_center_ps" in pars_dict else False
        )
        per_l_combine = (
            pars_dict["per_l_combine"] if "per_l_combine" in pars_dict else False
        )
        N_PSEUDO_SPECIES = (
            pars_dict["n_pseudo_species"] if "n_pseudo_species" in pars_dict else 4
        )

        self.hypers = {}
        if "hypers_ps" in self.model_parameters:
            self.hypers["spherical_expansion"] = self.model_parameters["hypers_ps"]
        if "hypers_rs" in self.model_parameters:
            self.hypers["radial_spectrum"] = self.model_parameters["hypers_rs"]

        dataset_grad = AtomisticDataset(
            frames,
            all_species,
            self.hypers,
            energies,
            forces,
            normalization=train_normalization,
            do_gradients=True,
        )

        dataloader_grad = create_dataloader(
            dataset_grad,
            batch_size=1,
            shuffle=False,
            device=device,
        )
        TORCH_REGULARIZER = 1e-2
        LINALG_REGULARIZER_ENERGIES = 1e-2
        LINALG_REGULARIZER_FORCES = 1e-1

        combiner = CombineSpecies(
            species=all_species,
            n_pseudo_species=N_PSEUDO_SPECIES,
            per_l_max=(
                self.hypers["spherical_expansion"]["max_angular"]
                if per_l_combine
                else 0
            ),
        )

        power_spectrum = CombinedPowerSpectrum(combiner)
        self.model = MultiBodyOrderModel(
            power_spectrum=power_spectrum,
            composition_regularizer=[1e-10],
            radial_spectrum_regularizer=[
                LINALG_REGULARIZER_ENERGIES,
                LINALG_REGULARIZER_FORCES,
            ],
            power_spectrum_regularizer=[
                LINALG_REGULARIZER_ENERGIES,
                LINALG_REGULARIZER_FORCES,
            ],
            optimizable_weights=True,
            random_initial_weights=True,
            ps_center_types=(all_species if per_center_ps else None),
        )

        self.model.to(device=device, dtype=torch.get_default_dtype())

        # initialize the model
        with torch.no_grad():
            for (
                composition,
                radial_spectrum,
                spherical_expansions,
                energies,
                forces,
            ) in dataloader_grad:
                # we want to intially train the model on all frames, to ensure the
                # support points come from the full dataset.
                self.model.initialize_model_weights(
                    composition, radial_spectrum, spherical_expansions, energies, forces
                )

        # modelfilename = "/home/lopanits/chemlearning/alchemical-learning/model_state_dict/CombinedLinearModel-4-mixed-100-train-100-test-4-max_ang-8-max_rad-0.3-sigma-4.0-cutoff-4-species-opt-weights-random-weights/1-epoch.pt"
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

        # Update ASE Atoms object (we only use ASE to handle any
        # re-wrapping of the atoms that needs to take place)
        self.atoms.set_cell(cell_matrix)
        self.atoms.set_positions(positions)

        # Compute representations and evaluate model
        ## TODO compute spherical expansion here
        ## TODO convert between numpy and pytroch
        frames = [self.atoms]
        energies = torch.tensor([[0.0]])
        forces = [torch.tensor(np.zeros((len(frames[0]), 3)))]
        HYPERS = self.hypers
        dataset_grad = AtomisticDataset(
            frames, all_species, self.hypers, energies, forces, do_gradients=True
        )

        dataloader_grad = create_dataloader(
            dataset_grad,
            batch_size=1,
            shuffle=False,
            device=device,
        )

        for (
            composition,
            radial_spectrum,
            spherical_expansions,
            energies,
            forces,
        ) in dataloader_grad:
            energy_torch, forces_torch = self.model(
                composition, radial_spectrum, spherical_expansions, forward_forces=True
            )
        energy = energy_torch.detach().numpy().flatten()
        forces = forces_torch.detach().numpy().flatten()
        stress_matrix = np.zeros((3, 3))
        # Symmetrize the stress matrix (replicate upper-diagonal entries)
        stress_matrix += np.triu(stress_matrix, k=1).T
        return energy, forces, stress_matrix
