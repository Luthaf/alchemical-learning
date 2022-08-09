# command-line version of the HEA-multibody notebook (with force learning)
import argparse
import copy
import json
import time
from datetime import datetime

import ase.io
import numpy as np
import torch

from mbmodel import CombinedPowerSpectrum, MultiBodyOrderModel
from utils.combine import CombineRadialSpecies, CombineSpecies
from utils.dataset import AtomisticDataset, create_dataloader

try:
    profile
except NameError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


torch.set_default_dtype(torch.float64)


def run_model(datafile, model_json, model_torch, device="cpu"):
    # --------- PARSING PARAMETERS --------- #
    pars_dict = json.load(open(model_json, "r"))
    force_weight = pars_dict["force_weight"]
    N_PSEUDO_SPECIES = pars_dict["n_pseudo_species"]
    prefix = pars_dict["prefix"]

    N_COMBINED = pars_dict["n_mixed_basis"] if "n_mixed_basis" in pars_dict else 0
    train_normalization = (
        pars_dict["normalization"] if "normalization" in pars_dict else None
    )
    do_restart = (
        pars_dict["restart"] if "restart" in pars_dict else True
    )  # defaults to restart if file present
    rng_seed = (
        pars_dict["seed"] if "seed" in pars_dict else None
    )  # defaults to random seed
    per_center_ps = (
        pars_dict["per_center_ps"] if "per_center_ps" in pars_dict else False
    )
    per_l_combine = (
        pars_dict["per_l_combine"] if "per_l_combine" in pars_dict else False
    )
    learning_rate = pars_dict["learning_rate"] if "learning_rate" in pars_dict else 0.05

    HYPERS_SMALL = pars_dict["hypers_ps"]
    if "radial_per_angular" in HYPERS_SMALL:
        HYPERS_SMALL["radial_per_angular"] = {
            int(k): v for k, v in HYPERS_SMALL["radial_per_angular"].items()
        }
    HYPERS_RADIAL = pars_dict["hypers_rs"]

    # --------- READING STUFF --------- #
    print("Reading file and properties")
    test_frames = ase.io.read(datafile, f"-10:")

    test_energies = (
        torch.tensor([frame.info["energy"] for frame in test_frames])
        .reshape(-1, 1)
        .to(dtype=torch.get_default_dtype())
    )
    test_forces = [
        torch.tensor(frame.arrays["forces"]).to(dtype=torch.get_default_dtype())
        for frame in test_frames
    ]

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

    print("Computing representations")
    test_dataset = AtomisticDataset(
        test_frames,
        all_species,
        {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion": HYPERS_SMALL},
        test_energies,
        test_forces,
        normalization=train_normalization,
    )

    print(test_frames[0].numbers)
    print("CHECK ", test_dataset.composition[0].block(0).values)
    print(test_dataset.radial_spectrum[0].block(0).values.sum(axis=1))
    print(test_dataset.spherical_expansions[0].block(2).values[0, 0])
    print(test_dataset.spherical_expansions[0].block(2).samples[0])

    # --------- DATA LOADERS --------- #
    print("Creating data loaders")
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=20,
        shuffle=False,
        device=device,
    )

    do_gradients = True
    print("Computing representations")
    test_dataset_grad = AtomisticDataset(
        test_frames,
        all_species,
        {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion": HYPERS_SMALL},
        test_energies,
        test_forces,
        normalization=train_normalization,
        do_gradients=True,
    )
    print(
        "sph grad",
        test_dataset_grad.spherical_expansions[0]
        .block(spherical_harmonics_l=2)
        .values[0, 0],
    )
    test_dataloader_grad = create_dataloader(
        test_dataset_grad,
        batch_size=2,
        shuffle=False,
        device=device,
    )

    # MODEL DEFINITION
    if N_COMBINED > 0:
        combiner = CombineRadialSpecies(
            n_species=len(all_species),
            max_radial=HYPERS_SMALL["max_radial"],
            n_combined_basis=N_COMBINED,
        )  # , per_l_max=(HYPERS_SMALL['max_angular'] if per_l_combine else 0))
    elif N_PSEUDO_SPECIES > 0:
        combiner = CombineSpecies(
            species=all_species,
            n_pseudo_species=N_PSEUDO_SPECIES,
            per_l_max=(HYPERS_SMALL["max_angular"] if per_l_combine else 0),
        )

    power_spectrum = CombinedPowerSpectrum(combiner)

    LINALG_REGULARIZER_ENERGIES = 1e-2
    LINALG_REGULARIZER_FORCES = 1e-1

    model = MultiBodyOrderModel(
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
        ) in test_dataloader:
            # we want to intially train the model on all frames, to ensure the
            # support points come from the full dataset.
            model.initialize_model_weights(
                composition,
                radial_spectrum,
                spherical_expansions,
                energies,
                forces,
                seed=rng_seed,
            )
            break

    del radial_spectrum, spherical_expansions
    npars = sum(len(p.detach().cpu().numpy().flatten()) for p in model.parameters())
    print(f"Model parameters: {npars}")

    try:
        state = torch.load(model_torch)
        print("Restarting model parameters from file")
        model.load_state_dict(state)
    except FileNotFoundError:
        print("Restart file not found")

    for p in model.power_spectrum.combiner.parameters():
        print("PS COMBINER ", p)
    with torch.no_grad():
        predicted = []
        reference = []
        f_predicted = []
        f_reference = []
        for (
            tcomposition,
            tradial_spectrum,
            tspherical_expansions,
            tenergies,
            tforces,
        ) in test_dataloader_grad:
            reference.append(tenergies)
            tpredicted_e, tpredicted_f = model(
                tcomposition,
                tradial_spectrum,
                tspherical_expansions,
                forward_forces=do_gradients,
            )
            print("comp", tcomposition.block(0).values[0])
            print(tspherical_expansions.block(2).samples[0])
            print(
                "sph", tspherical_expansions.block(spherical_harmonics_l=2).values[0, 0]
            )
            comb = model.power_spectrum.combiner(tspherical_expansions)
            print("comb", comb.block(spherical_harmonics_l=2).values[0, 0])
            print("samp", tspherical_expansions.block(spherical_harmonics_l=2).samples)
            pwr = model.power_spectrum(tspherical_expansions)
            print("PWR:", pwr.keys[0], pwr.block(0).values[0])
            print("samp", pwr.block(0).samples)
            print(tenergies, tpredicted_e)
            predicted.append(tpredicted_e)
            if do_gradients:
                f_predicted.append(tpredicted_f)
                f_reference.append(tforces)

        reference = torch.vstack(reference)
        predicted = torch.vstack(predicted)
        output.write(f"{n_epochs_total} {loss} {test_mae}")
        if do_gradients:
            f_reference = torch.vstack(f_reference)
            f_predicted = torch.vstack(f_predicted)
            f_test_mae = loss_mae(f_predicted, f_reference) / n_test / 42
            output.write(f"{f_test_mae}")
        output.write("\n")
        output.flush()
    all_tests.append(test_mae.item())
    f_all_tests.append(f_test_mae.item())
    print(
        f"epoch {n_epochs_total} took {epoch_time:.4}s, optimizer loss={loss:.4}, test mae={test_mae:.4}"
        + (f" test mae force={f_test_mae:.4}")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This tool fits a potential for a multi-component system using an alchemical mixture model.
        Usage:
             python multibody-force.py datafile.xyz parameters_file [-d device]
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
    run_model(args.datafile, args.model_json, args.model_torch)
