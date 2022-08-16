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


def run_fit(datafile, parameters, device="cpu"):
    # --------- PARSING PARAMETERS --------- #
    pars_dict = json.load(open(parameters, "r"))
    n_epochs = pars_dict["n_epochs"]
    n_test = pars_dict["n_test"]
    n_train = pars_dict["n_train"]
    n_train_forces = pars_dict["n_train_forces"]
    force_weight = pars_dict["force_weight"]
    N_PSEUDO_SPECIES = pars_dict["n_pseudo_species"]
    prefix = pars_dict["prefix"]

    N_COMBINED = pars_dict["n_mixed_basis"] if "n_mixed_basis" in pars_dict else 0
    nn_layer_size = (
        pars_dict["layer_size"] if "layer_size" in pars_dict else 0
    )  # defaults to restart if file present
    
    train_normalization = (
        pars_dict["normalization"] if "normalization" in pars_dict else None
    )
    do_restart = (
        pars_dict["restart"] if "restart" in pars_dict else True
    )  # defaults to restart if file present
    n_epochs_total = pars_dict["n_epochs_total"] if "n_epochs_total" in pars_dict else 0
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
    frames = ase.io.read(datafile, f":{n_test + n_train + n_train_forces}")

    train_frames = frames[:n_train]
    train_forces_frames = frames[n_train : n_train + n_train_forces]
    test_frames = frames[-n_test:]

    train_energies = (
        torch.tensor([frame.info["energy"] for frame in train_frames])
        .reshape(-1, 1)
        .to(dtype=torch.get_default_dtype())
    )

    test_energies = (
        torch.tensor([frame.info["energy"] for frame in test_frames])
        .reshape(-1, 1)
        .to(dtype=torch.get_default_dtype())
    )

    train_forces_e = (
        torch.tensor([frame.info["energy"] for frame in train_forces_frames])
        .reshape(-1, 1)
        .to(dtype=torch.get_default_dtype())
    )

    train_forces_f = [
        torch.tensor(frame.arrays["forces"]).to(dtype=torch.get_default_dtype())
        for frame in train_forces_frames
    ]

    test_forces = [
        torch.tensor(frame.arrays["forces"]).to(dtype=torch.get_default_dtype())
        for frame in test_frames
    ]

    print(f"using {n_train} training frames and {n_train_forces} force frames")

    all_species = set()
    for frame in frames:
        all_species.update(frame.numbers)

    all_species = list(map(lambda u: int(u), all_species))
    print("All species ", all_species)

    print("Computing representations")
    train_dataset = AtomisticDataset(
        train_frames,
        all_species,
        {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion": HYPERS_SMALL},
        train_energies,
        normalization=train_normalization,
    )
    if train_normalization is not None:
        train_normalization = {
            "radial_spectrum": train_dataset.radial_norm,
            "spherical_expansion": train_dataset.spherical_expansion_norm,
        }

        # also overwrites the option, so that this will be saved in the .json
        # parameter file
        pars_dict["normalization"] = train_normalization
    train_forces_dataset = AtomisticDataset(
        train_forces_frames,
        all_species,
        {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion": HYPERS_SMALL},
        train_forces_e,
        normalization=train_normalization,
    )
    test_dataset = AtomisticDataset(
        test_frames,
        all_species,
        {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion": HYPERS_SMALL},
        test_energies,
        normalization=train_normalization,
    )

    # do gradients, we need them to at least check on the test set. set
    # force_weight to zero to cut it off
    do_gradients = True
    if do_gradients is True:
        print("Computing data with gradients")
        HYPERS_GRAD = copy.deepcopy(HYPERS_SMALL)
        HYPERS_RAD_GRAD = copy.deepcopy(HYPERS_RADIAL)
        train_forces_dataset_grad = AtomisticDataset(
            train_forces_frames,
            all_species,
            {"radial_spectrum": HYPERS_RAD_GRAD, "spherical_expansion": HYPERS_GRAD},
            train_forces_e,
            train_forces_f,
            normalization=train_normalization,
            do_gradients=True,
        )
        test_dataset_grad = AtomisticDataset(
            test_frames,
            all_species,
            {"radial_spectrum": HYPERS_RAD_GRAD, "spherical_expansion": HYPERS_GRAD},
            test_energies,
            test_forces,
            normalization=train_normalization,
            do_gradients=True,
        )
    else:
        train_forces_dataset_grad = train_forces_dataset
        test_dataset_grad = test_dataset

    # --------- DATA LOADERS --------- #
    print("Creating data loaders")
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=200,
        shuffle=True,
        device=device,
    )

    train_forces_dataloader = create_dataloader(
        train_forces_dataset,
        batch_size=200,
        shuffle=True,
        device=device,
    )

    train_dataloader_no_batch = create_dataloader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        device=device,
    )

    train_forces_dataloader_no_batch = create_dataloader(
        train_forces_dataset,
        batch_size=len(train_forces_dataset),
        shuffle=False,
        device=device,
    )

    train_forces_dataloader_single_frame = create_dataloader(
        train_forces_dataset,
        batch_size=1,
        shuffle=False,
        device=device,
    )

    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=50,
        shuffle=False,
        device=device,
    )

    if do_gradients is True:
        train_forces_dataloader_grad = create_dataloader(
            train_forces_dataset_grad,
            batch_size=50,
            shuffle=True,
            device=device,
        )

        train_forces_dataloader_grad_single_frame = create_dataloader(
            train_forces_dataset_grad,
            batch_size=1,
            shuffle=False,
            device=device,
        )

        train_forces_dataloader_grad_no_batch = create_dataloader(
            train_forces_dataset_grad,
            batch_size=len(train_forces_dataset_grad),
            shuffle=False,
            device=device,
        )

        test_dataloader_grad = create_dataloader(
            test_dataset_grad,
            batch_size=50,
            shuffle=False,
            device=device,
        )
    else:
        train_forces_dataloader_grad = train_forces_dataloader
        train_forces_dataloader_grad_single_frame = train_forces_dataloader_single_frame
        train_forces_dataloader_grad_no_batch = train_forces_dataloader_no_batch
        test_dataloader_grad = test_dataloader

    def loss_mae(predicted, actual):
        return torch.sum(torch.abs(predicted.flatten() - actual.flatten()))

    def loss_mse(predicted, actual):
        return torch.sum((predicted.flatten() - actual.flatten()) ** 2)

    def loss_rmse(predicted, actual):
        return np.sqrt(loss_mse(predicted, actual))

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

    TORCH_REGULARIZER_COMPOSITION = pars_dict["regularizer_x"] if "regularizer_x" in pars_dict else None
    TORCH_REGULARIZER_RADIAL_SPECTRUM = pars_dict["regularizer_rs"]  if "regularizer_rs" in pars_dict else None
    TORCH_REGULARIZER_POWER_SPECTRUM = pars_dict["regularizer_ps"]  if "regularizer_ps" in pars_dict else None
    TORCH_REGULARIZER_NN = pars_dict["regularizer_nn"] if "regularizer_nn" in pars_dict else None

    model = MultiBodyOrderModel(
        power_spectrum=power_spectrum,
        composition_regularizer=TORCH_REGULARIZER_COMPOSITION,
        radial_spectrum_regularizer=TORCH_REGULARIZER_RADIAL_SPECTRUM,
        power_spectrum_regularizer=TORCH_REGULARIZER_POWER_SPECTRUM,
        optimizable_weights=True,
        random_initial_weights=True,
        nn_layer_size=nn_layer_size,
        ps_center_types=(all_species if per_center_ps else None),
    )

    model.to(device=device, dtype=torch.get_default_dtype())

    if model.random_initial_weights:
        dataloader_initialization = train_forces_dataloader_grad_single_frame
    else:
        dataloader_initialization = train_dataloader_no_batch

    # --------- INITIALIZE MODEL --------- #
    print("Initializing model")
    with torch.no_grad():
        for (
            composition,
            radial_spectrum,
            spherical_expansions,
            energies,
            forces,
        ) in dataloader_initialization:
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

    now = datetime.now().strftime("%y%m%d-%H%M")
    filename = f"{prefix}"

    json.dump(pars_dict, open(f"{filename}_{now}.json", "w"))

    if do_restart:
        try:
            state = torch.load(f"{filename}-restart.torch")
            print("Restarting model parameters from file")
            model.load_state_dict(state)
        except FileNotFoundError:
            print("Restart file not found")

    lr = learning_rate
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, line_search_fn="strong_wolfe", history_size=128
    )

    all_losses = []
    all_tests = []
    f_all_tests = []

    output = open(f"{filename}_{now}.dat", "w")
    output.write("# epoch  train_loss  test_mae test_mae_f\n")
    torch.save(model.state_dict(), f"{filename}_{now}-init.torch")

    assert model.optimizable_weights
    high_mem = True
    if high_mem:
        composition, radial_spectrum, spherical_expansions, energies, forces = next(
            iter(train_dataloader_no_batch)
        )
        del train_dataset
        (
            f_composition,
            f_radial_spectrum,
            f_spherical_expansions,
            f_energies,
            f_forces,
        ) = next(iter(train_forces_dataloader_grad_no_batch))
        del train_forces_dataset
    best_mae = 1e100
    for epoch in range(n_epochs_total, n_epochs):
        print("Beginning epoch", epoch)
        epoch_start = time.time()

        @profile
        def single_step(
            composition=composition,
            radial_spectrum=radial_spectrum,
            spherical_expansions=spherical_expansions,
            energies=energies,
            forces=forces,
        ):
            # global composition, radial_spectrum, spherical_expansions, energies
            optimizer.zero_grad()
            loss = torch.zeros(size=(1,), device=device)
            loss_force = torch.zeros(size=(1,), device=device)
            if high_mem:
                predicted, _ = model(
                    composition,
                    radial_spectrum,
                    spherical_expansions,
                    forward_forces=False,
                )
                loss += loss_mse(predicted, energies)
                f_predicted_e, f_predicted_f = model(
                    f_composition,
                    f_radial_spectrum,
                    f_spherical_expansions,
                    forward_forces=do_gradients,
                )
                loss += loss_mse(f_predicted_e, f_energies)
                if do_gradients:
                    loss_force += loss_mse(f_predicted_f, f_forces) / 42
            else:
                for (
                    composition,
                    radial_spectrum,
                    spherical_expansions,
                    energies,
                    forces,
                ) in train_dataloader:
                    predicted, _ = model(
                        composition,
                        radial_spectrum,
                        spherical_expansions,
                        forward_forces=False,
                    )
                    loss += loss_mse(predicted, energies)
                raise ValueError("MUST IMPLEMENT FORCE CALCULATOR FOR THIS PATH!")
            loss /= n_train + n_train_forces
            loss_force /= n_train_forces

            if model.composition_model is not None:
                loss += TORCH_REGULARIZER_COMPOSITION * torch.linalg.norm(
                    model.composition_model.weights
                )
            if model.radial_spectrum_model is not None:
                loss += TORCH_REGULARIZER_RADIAL_SPECTRUM * torch.linalg.norm(
                    model.radial_spectrum_model.weights
                )
            if model.power_spectrum_model is not None:
                loss += TORCH_REGULARIZER_POWER_SPECTRUM * torch.linalg.norm(
                    model.power_spectrum_model.weights
                )
            if model.nn_model is not None:
                loss += TORCH_REGULARIZER_NN * sum((p**2).sum() for p in model.nn_model.parameters())

            print(
                f"Train loss: {(loss+loss_force*force_weight).item()} E={loss.item()}, F={loss_force.item()}"
            )
            loss += loss_force * force_weight
            loss.backward(retain_graph=False)
            return loss

        loss = optimizer.step(single_step)
        loss = loss.item()
        all_losses.append(loss)

        epoch_time = time.time() - epoch_start
        if epoch % 1 == 0:
            print(
                "norms",
                np.linalg.norm(
                    0
                    if model.composition_model is None
                    else model.composition_model.weights.detach().cpu().numpy()
                ),
                np.linalg.norm(
                    0
                    if model.radial_spectrum_model is None
                    else model.radial_spectrum_model.weights.detach().cpu().numpy()
                ),
                np.linalg.norm(
                    0
                    if model.power_spectrum_model is None
                    else model.power_spectrum_model.weights.detach().cpu().numpy()
                ),
            )
            print(
                "gradients",
                np.linalg.norm(
                    0
                    if model.composition_model is None
                    else model.composition_model.weights.grad.detach().cpu().numpy()
                ),
                np.linalg.norm(
                    0
                    if model.radial_spectrum_model is None
                    else model.radial_spectrum_model.weights.grad.detach().cpu().numpy()
                ),
                np.linalg.norm(
                    0
                    if model.power_spectrum_model is None
                    else model.power_spectrum_model.weights.grad.detach().cpu().numpy()
                ),
            )
            if True:  #with torch.no_grad(): ### this interferes with the autograd force calculator
                # train set errors
                if high_mem:
                    predicted, _ = model(
                        composition,
                        radial_spectrum,
                        spherical_expansions,
                        forward_forces=False,
                    )   
                    train_mae = loss_mae(predicted, energies)
                    f_predicted_e, f_predicted_f = model(
                        f_composition,
                        f_radial_spectrum,
                        f_spherical_expansions,
                        forward_forces=do_gradients,
                    )
                    train_mae += loss_mae(f_predicted_e, f_energies)                    
                    train_mae /= n_train + n_train_forces
                    if do_gradients:
                        f_train_mae = loss_mae(f_predicted_f, f_forces) / (3*42) / n_train_forces
                    print("Energy component stats (mean absolute values over last test batch)")
                    for k in model.latest_energies:
                        print(k, ":  ", model.latest_energies[k].detach().abs().mean().item())
                    print("Energy component stats (mean absolute values over last test batch)")
                    for k in model.latest_forces:
                        print(k, ":  ", model.latest_forces[k].detach().abs().mean().item())
                else:
                    train_mae, f_train_mae = 0,0

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
                    predicted.append(tpredicted_e)
                    if do_gradients:
                        f_predicted.append(tpredicted_f)
                        f_reference.append(tforces)

                reference = torch.vstack(reference)
                predicted = torch.vstack(predicted)
                test_mae = loss_mae(predicted, reference) / n_test
                output.write(f"{n_epochs_total} {loss} {test_mae}")
                if do_gradients:
                    f_reference = torch.vstack(f_reference)
                    f_predicted = torch.vstack(f_predicted)
                    f_test_mae = loss_mae(f_predicted, f_reference) / n_test / (3*42)
                    output.write(f"{f_test_mae}")
                output.write("\n")
                output.flush()
            all_tests.append(test_mae.item())
            f_all_tests.append(f_test_mae.item())
            print(
                f"epoch {n_epochs_total} took {epoch_time:.4}s, optimizer loss={loss:.4}, test mae={test_mae:.4}"
                + (f" test mae force={f_test_mae:.4}") + (f" train mae={train_mae:.4}") + 
                (f" train mae force={f_train_mae:.4}")
            )
            with open(f"{filename}_epochs.dat", "a") as f:
                np.savetxt(f, [[n_epochs_total, test_mae.detach(), f_test_mae.detach(), train_mae.detach(), f_train_mae.detach()]])  

            np.savetxt(
                f"{filename}-energy_test.dat",
                np.hstack([reference.cpu().detach().numpy(), predicted.cpu().detach().numpy()]),
            )
            np.savetxt(
                f"{filename}-force_test.dat",
                np.hstack(
                    [
                        f_reference.cpu().detach().numpy().reshape(-1, 1),
                        f_predicted.cpu().detach().numpy().reshape(-1, 1),
                    ]
                ),
            )
            if test_mae < best_mae:
                best_mae = test_mae.detach().item()
                torch.save(
                    model.state_dict(), f"{filename}-best.torch"
                )  # no NOW string so we can restart but keep track of initial and final files
            del train_mae, f_train_mae, test_mae, f_test_mae



        del loss
        n_epochs_total += 1
        torch.save(
            model.state_dict(), f"{filename}-restart.torch"
        )  # no NOW string so we can restart but keep track of initial and final files

    torch.save(model.state_dict(), f"{filename}_{now}-final.torch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This tool fits a potential for a multi-component system using an alchemical mixture model.
        Usage:
             python multibody-force.py datafile.xyz parameters_file [-d device]
        """
    )

    parser.add_argument(
        "datafile",
        type=str,
        help="The path to the structure file that contains reference energies and forces (extended xyz format)",
    )
    parser.add_argument(
        "parameters",
        type=str,
        help="The file containing the parameters. JSON formatted dictionary.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The torch device to run the model on.",
    )
    args = parser.parse_args()
    run_fit(args.datafile, args.parameters, args.device)
