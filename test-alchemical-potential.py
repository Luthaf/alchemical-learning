import argparse
import json
import os
import sys
import time
from datetime import datetime

import ase.io
import numpy as np
import torch

from utils.combine import CombineSpecies
from utils.dataset import AtomisticDataset, create_dataloader
from utils.model import AlchemicalModel

try:
    profile
except NameError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


torch.set_default_dtype(torch.float64)


def loss_mae(predicted, actual):
    return torch.sum(torch.abs(predicted.flatten() - actual.flatten()))


def loss_mse(predicted, actual):
    return torch.sum((predicted.flatten() - actual.flatten()) ** 2)


def loss_rmse(predicted, actual):
    return np.sqrt(loss_mse(predicted, actual))


def extract_energy_forces(frames):
    energies = (
        torch.tensor([ (frame.info["energy"] if "energy" in frame.info else 0) for frame in frames])
        .reshape(-1, 1)
        .to(dtype=torch.get_default_dtype())
    )

    forces = [
        torch.tensor(
        frame.arrays["forces"] if "forces" in frame.arrays else frame.positions*0.0        
        ).to(dtype=torch.get_default_dtype())
        for frame in frames
    ]

    return energies, forces


def main(datafile, parameters, weights, device="cpu"):
    with open(parameters) as fd:
        parameters = json.load(fd)

    # --------- READING STUFF --------- #
    print("Reading file and properties")
    all_species = parameters.get("species", [])
    if len(all_species) == 0:
        print("Warning: A list of species is not found! Please, set it in the json file!")

    frames = ase.io.read(datafile, ":")
    all_species_from_data = set()
    for frame in frames:
        all_species_from_data.update(frame.numbers)
    all_species_from_data = list(map(lambda u: int(u), all_species_from_data))

    if set(all_species_from_data).issubset(all_species):
        print("All species ", all_species)
    else:
        print("Warning: Not all species from data are found in the json file!")
        print("Please, set a new list of species in the json file!")
        print("Try to put the following line in the json file:")
        print("\t\"species\": ", all_species_from_data, ",", sep='')
        sys.exit(1)

    test_frames = frames
    test_energies, test_forces = extract_energy_forces(test_frames)

    print("Computing representations")
    hypers_ps = parameters["hypers_ps"]
    if "radial_per_angular" in hypers_ps:
        hypers_ps["radial_per_angular"] = {
            int(k): v for k, v in hypers_ps["radial_per_angular"].items()
        }
    hypers_rs = parameters.get("hypers_rs")

    test_dataset_grad = AtomisticDataset(
        test_frames,
        all_species,
        {"radial_spectrum": hypers_rs, "spherical_expansion": hypers_ps},
        test_energies,
        test_forces,
        do_gradients=True,
    )

    # --------- DATA LOADERS --------- #
    print("Creating data loaders")
    test_dataloader_grad = create_dataloader(
        test_dataset_grad,
        batch_size=5,
        shuffle=False,
        device=device,
    )

    combiner = CombineSpecies(
        species=all_species,
        n_pseudo_species=parameters["n_pseudo_species"],
        # TODO: remove this from code
        per_l_max=0,
    )

    COMPOSITION_REGULARIZER = parameters.get("composition_regularizer")
    RADIAL_SPECTRUM_REGULARIZER = parameters.get("radial_spectrum_regularizer")
    POWER_SPECTRUM_REGULARIZER = parameters.get("power_spectrum_regularizer")
    NN_REGULARIZER = parameters.get("nn_regularizer")
    FORCES_LOSS_WEIGHT = parameters.get("forces_loss_weight")

    model = AlchemicalModel(
        combiner=combiner,
        composition_regularizer=COMPOSITION_REGULARIZER,
        radial_spectrum_regularizer=RADIAL_SPECTRUM_REGULARIZER,
        power_spectrum_regularizer=POWER_SPECTRUM_REGULARIZER,
        nn_layer_size=parameters.get("nn_layer_size"),
        # TODO: remove from code?
        ps_center_types=None,
    )

    model.to(device=device, dtype=torch.get_default_dtype())

    # --------- INITIALIZE MODEL --------- #
    print("Initializing model")
    with torch.no_grad():
        for (
            composition,
            radial_spectrum,
            spherical_expansions,
            energies,
            forces,
        ) in test_dataloader_grad:
            model.initialize_model_weights(
                composition,
                radial_spectrum,
                spherical_expansions,
                energies,
                forces,
                seed=parameters.get("seed"),
            )
            break

    del radial_spectrum, spherical_expansions
    n_parameters = sum(
        len(p.detach().cpu().numpy().flatten()) for p in model.parameters()
    )
    print(f"Model parameters: {n_parameters}")
    model.load_state_dict(torch.load(weights))

    
    predicted = []
    reference = []
    predicted_forces = []
    reference_forces = []
    test_base = 0
    for (
        test_composition,
        test_radial_spectrum,
        test_spherical_expansions,
        test_energies,
        test_forces,
    ) in test_dataloader_grad:
        reference.append(test_energies)
        test_predicted_e, test_predicted_f = model(
            test_composition,
            test_radial_spectrum,
            test_spherical_expansions,
            forward_forces=True,
        )
        
        predicted.append(test_predicted_e.detach())
        predicted_forces.append(test_predicted_f.detach())
        reference_forces.append(test_forces)
        for i, pe in enumerate(test_predicted_e):
            pf = test_predicted_f[:len(test_frames[test_base+i])]
            test_predicted_f= test_predicted_f[len(test_frames[test_base+i]):]
            test_frames[test_base+i].info["predicted_energy"] = pe.item()
            test_frames[test_base+i].arrays["predicted_forces"] = pf.detach().numpy()
        test_base += len(test_predicted_e)
        
    reference = torch.vstack(reference)
    predicted = torch.vstack(predicted)
    test_mae = loss_mae(predicted, reference) / len(test_energies)

    reference_forces = torch.vstack(reference_forces)
    predicted_forces = torch.vstack(predicted_forces)
    # TODO: do this properly
    test_mae_forces = (
        loss_mae(predicted_forces, reference_forces) / len(test_forces) / (3 * 42)
    )
    
    print("overall test MAE", test_mae, " forces ", test_mae_forces)
    
    ase.io.write("predicted_"+datafile, test_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"""
        This tool runs a potential for a multi-component system on a given data file

        Usage:
             python {sys.argv[0]} datafile.xyz parameters.json model.torch [-d device]
        """
    )

    parser.add_argument(
        "datafile",
        type=str,
        help="file containing reference energies and forces in extended xyz format",
    )
    parser.add_argument(
        "parameters",
        type=str,
        help="file containing the parameters in JSON format",
    )
    parser.add_argument(
        "model",
        type=str,
        help="torch file containing the model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device to run the model on",
    )
    args = parser.parse_args()

    main(args.datafile, args.parameters, args.model, args.device)
