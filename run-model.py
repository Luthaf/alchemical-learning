"""Generic calculator-style interface for MD"""
import ase.io
import argparse
import numpy as np
import json

import torch
from utils.dataset import AtomisticDataset, create_dataloader
from driver import GenericMDCalculator, all_species

torch.set_default_dtype(torch.float64)
device = "cpu"


class ModelCalculator(GenericMDCalculator):
    def calculate(self, frame):

        energies = torch.tensor([[0.0]])
        forces = [torch.tensor(np.zeros((len(frame), 3)))]
        HYPERS = self.hypers
        dataset_grad = AtomisticDataset(
            [frame], all_species, self.hypers, energies, forces, do_gradients=True
        )

        print("norm:", dataset_grad.radial_norm, dataset_grad.spherical_expansion_norm)
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


def run_model(model_json, model_torch, datafile):
    calculator = ModelCalculator(model_torch, model_json, True, datafile)
    print("Reading file...")
    frames = ase.io.read(datafile, ":100")
    print("Computing frames...")
    for f in frames:
        energy, forces, stress = calculator.calculate(f)
        print(energy, f.info["energy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This runs a fitted model on a given target file
        Usage:
             python model.json model.torch datafile.xyz
        """
    )

    parser.add_argument(
        "model_json",
        type=str,
        help="The file containing the model settings. JSON formatted dictionary.",
    )
    parser.add_argument(
        "model_torch",
        type=str,
        help="The file containing the model parameters. Torch model dump.",
    )
    parser.add_argument(
        "datafile",
        type=str,
        help="The path to the structure file that contains target structures (extended xyz format)",
    )
    args = parser.parse_args()
    run_model(args.model_json, args.model_torch, args.datafile)
