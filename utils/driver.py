"""Generic calculator-style interface for MD"""
import ase.io
import numpy as np
import json

import torch
from .dataset import AtomisticDataset, create_dataloader
from .soap import PowerSpectrum
from .combine import CombineSpecies, CombineRadial, CombineRadialSpecies
from .combined_linear import CombinedLinearModel
from .linear import LinearModel
from .operations import SumStructures, remove_gradient
torch.set_default_dtype(torch.float64)

HYPERS_SMALL = {
        "cutoff": 4.0,
        "max_angular": 4,
        "max_radial": 8,
        "center_atom_weight":1, 
        "atomic_gaussian_width": 0.3,
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "radial_basis": {"SplinedGto": {"accuracy": 1e-6}},
        "gradients": True,
        # TODO: implement this in rascaline itself
        "radial_per_angular": {
            # l: n
            0: 10,
            1: 8,
            2: 8,
            3: 4,
            4: 4,
        }
        }

device = 'cpu'



all_species = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 44,
 45, 46, 47, 71, 72, 73, 74, 77, 78, 79]

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
        self, model_path, hypers_path, is_periodic, structure_template=None, atomic_numbers=None
    ):
        super().__init__()
        #self.model = torch.load(model_pt)
        # Structure initialization
        self.is_periodic = is_periodic
        self.hypers = json.load(open(hypers_path),  object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
        

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
        energies = torch.tensor([[0.]])
        forces =  [torch.tensor(np.zeros((len(frames[0]), 3)))]
        HYPERS = self.hypers
        dataset_grad = AtomisticDataset(frames, all_species, HYPERS, energies, forces)
        
        dataloader_grad = create_dataloader(
        dataset_grad,
        batch_size=1,
        shuffle=False,
        device=device,
        )

        N_PSEUDO_SPECIES = 4
        TORCH_REGULARIZER = 1e-2
        LINALG_REGULARIZER_ENERGIES = 1e-2
        LINALG_REGULARIZER_FORCES = 1e-1
        N_COMBINED_RADIAL = 4
        COMBINER = "species" #  "sequential" # "radial_species"
        if COMBINER=="species":
            combiner = CombineSpecies(species=all_species, n_pseudo_species=N_PSEUDO_SPECIES)
        self.model = CombinedLinearModel(
        combiner=combiner, 
        regularizer=[LINALG_REGULARIZER_ENERGIES, LINALG_REGULARIZER_FORCES],
        optimizable_weights=True,
        random_initial_weights=True,
        )
        self.model.to(device=device, dtype=torch.get_default_dtype())

        # initialize the model
        with torch.no_grad():
            for _, _, spherical_expansions, energies, forces in dataloader_grad:
                # we want to intially train the model on all frames, to ensure the
                # support points come from the full dataset.
                self.model.initialize_model_weights(spherical_expansions, energies)

        del spherical_expansions

        if self.model.optimizable_weights:
            torch_loss_regularizer = TORCH_REGULARIZER
        else:
            torch_loss_regularizer = 0
            # we can not use batches if we are training with linear algebra, we need to
            # have all training frames available
            assert train_dataloader.batch_size >= len(train_frames)
        
        #modelfilename = "/home/lopanits/chemlearning/alchemical-learning/model_state_dict/CombinedLinearModel-4-mixed-100-train-100-test-4-max_ang-8-max_rad-0.3-sigma-4.0-cutoff-4-species-opt-weights-random-weights/1-epoch.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=device))
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
        energies = torch.tensor([[0.]])
        forces =  [torch.tensor(np.zeros((len(frames[0]), 3)))]
        HYPERS =  self.hypers
        dataset_grad = AtomisticDataset(frames, all_species, HYPERS, energies, forces)
        
        dataloader_grad = create_dataloader(
        dataset_grad,
        batch_size=1,
        shuffle=False,
        device=device,
        )
        
        for _, _, spherical_expansions, energies, forces in dataloader_grad:
            energy_torch, forces_torch = self.model(spherical_expansions, forward_forces=True)
        energy = energy_torch.detach().numpy().flatten()
        forces = forces_torch.detach().numpy().flatten()
        stress_matrix = np.zeros((3, 3))
        # Symmetrize the stress matrix (replicate upper-diagonal entries)
        stress_matrix += np.triu(stress_matrix, k=1).T
        return energy, forces, stress_matrix
