import argparse
import json
import os
import sys
from time import strftime
from time import gmtime
import time
from datetime import datetime

import ase.io
import numpy as np
import torch

from utils.combine import CombineSpecies, CombineRadialSpecies, CombineRadialSpeciesWithAngular, \
    CombineRadialSpeciesWithAngularAdaptBasis, CombineRadialSpeciesWithAngularAdaptBasisRadial, \
    CombineRadialSpeciesWithCentralSpecies, CombineSpeciesWithCentralSpecies
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
        torch.tensor([frame.info["energy"] for frame in frames])
        .reshape(-1, 1)
        .to(dtype=torch.get_default_dtype())
    )

    forces = [
        torch.tensor(frame.arrays["forces"]).to(dtype=torch.get_default_dtype())
        for frame in frames
    ]

    return energies, forces


def main(datafile, parameters, device="cpu"):
    with open(parameters) as fd:
        parameters = json.load(fd)

    # --------- READING STUFF --------- #
    print("Reading file and properties")
    n_test = parameters["n_test"]
    n_train = parameters["n_train"]
    n_train_forces = parameters["n_train_forces"]
    do_gradients = parameters.get("do_gradients", True)
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
        return

    train_frames = frames[:n_train]
    train_forces_frames = frames[n_train : n_train + n_train_forces]
    test_frames = frames[-n_test:]

    train_energies, _ = extract_energy_forces(train_frames)
    train_f_energies, train_f_forces = extract_energy_forces(train_forces_frames)
    test_energies, test_forces = extract_energy_forces(test_frames)

    print(
        f"using {n_train} training frames and {n_train_forces} training "
        "frames with forces"
    )

    print("Computing representations")
    hypers_ps = parameters["hypers_ps"].copy()
    
    if "radial_per_angular" in hypers_ps:
        hypers_ps["radial_per_angular"] = {
            int(k): v for k, v in hypers_ps["radial_per_angular"].items()
        }
    combined_basis_per_angular = None
    if "combined_basis_per_angular" in hypers_ps:
        combined_basis_per_angular = hypers_ps.pop("combined_basis_per_angular")
        combined_basis_per_angular = {
            int(k): v for k, v in combined_basis_per_angular.items()
        }

    
    hypers_rs = parameters.get("hypers_rs")
    species_center_key_to_samples = parameters.get("species_center_key_to_samples", True)

    train_dataset = AtomisticDataset(
        train_frames,
        all_species,
        {"radial_spectrum": hypers_rs, "spherical_expansion": hypers_ps},
        train_energies,
        species_center_key_to_samples=species_center_key_to_samples,
    )

    if n_train_forces != 0:
        train_forces_dataset_grad = AtomisticDataset(
            train_forces_frames,
            all_species,
            {"radial_spectrum": hypers_rs, "spherical_expansion": hypers_ps},
            train_f_energies,
            train_f_forces,
            do_gradients=True,
            species_center_key_to_samples=species_center_key_to_samples,
        )
    else:
        train_forces_dataset_grad = None

    test_dataset = AtomisticDataset(
        test_frames,
        all_species,
        {"radial_spectrum": hypers_rs, "spherical_expansion": hypers_ps},
        test_energies,
        test_forces,
        do_gradients=do_gradients,
        species_center_key_to_samples=species_center_key_to_samples,
    )

    # --------- DATA LOADERS --------- #
    print("Creating data loaders")
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=parameters.get("batch_size", len(train_dataset)),
        shuffle=False,
        device=device,
    )
    train_dataloader_no_batch = create_dataloader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        device=device,
    )
    train_dataloader_single_frame = create_dataloader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        device=device,
    )

    if n_train_forces != 0:
        train_forces_dataloader_grad_no_batch = create_dataloader(
            train_forces_dataset_grad,
            batch_size=len(train_forces_dataset_grad),
            shuffle=False,
            device=device,
        )
        train_forces_dataloader_grad = create_dataloader(
            train_forces_dataset_grad,
            batch_size=parameters.get("batch_size", len(train_forces_dataset_grad)),
            shuffle=False,
            device=device,
        )
    else:
        train_forces_dataloader_grad_no_batch = None
        train_forces_dataloader_grad = None

    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=50,
        shuffle=False,
        device=device,
    )
    
    COMBINER_TYPE = parameters.get("combiner", "CombineRadialSpecies")
    assert (species_center_key_to_samples == False) == (COMBINER_TYPE in ["CombineRadialSpeciesWithCentralSpecies", "CombineSpeciesWithCentralSpecies"])

    combiner = None
    if COMBINER_TYPE == "CombineSpecies":
        combiner = CombineSpecies(
            species=all_species,
            n_pseudo_species=parameters["n_pseudo_species"],
            # TODO: remove this from code
            per_l_max=0
        )
    elif COMBINER_TYPE == "CombineRadialSpecies":
        combiner = CombineRadialSpecies(
            n_species=len(all_species),
            max_radial=hypers_ps["max_radial"],
            n_combined_basis=parameters.get("n_combined_basis", 16),
            seed=parameters.get("seed")
        )
    elif COMBINER_TYPE == "CombineRadialSpeciesWithAngular":
        combiner = CombineRadialSpeciesWithAngular(
            n_species=len(all_species),
            max_radial=hypers_ps["max_radial"],
            max_angular=hypers_ps["max_angular"],
            n_combined_basis=parameters.get("n_combined_basis", 16),
            seed=parameters.get("seed")
        )
    elif COMBINER_TYPE == "CombineRadialSpeciesWithAngularAdaptBasis":
        combiner = CombineRadialSpeciesWithAngularAdaptBasis(
            n_species=len(all_species),
            max_radial=hypers_ps["max_radial"],
            max_angular=hypers_ps["max_angular"],
            n_combined_basis=parameters.get("n_combined_basis", 16),
            combined_basis_per_angular=combined_basis_per_angular,
            seed=parameters.get("seed")
        )
    elif COMBINER_TYPE == "CombineRadialSpeciesWithAngularAdaptBasisRadial":
        combiner = CombineRadialSpeciesWithAngularAdaptBasisRadial(
            n_species=len(all_species),
            max_radial=hypers_ps["max_radial"],
            max_angular=hypers_ps["max_angular"],
            radial_per_angular=hypers_ps["radial_per_angular"],
            n_combined_basis=parameters.get("n_combined_basis", 16),
            combined_basis_per_angular=combined_basis_per_angular,
            seed=parameters.get("seed")
        )
    elif COMBINER_TYPE == "CombineRadialSpeciesWithCentralSpecies":
        assert species_center_key_to_samples == False
        combiner = CombineRadialSpeciesWithCentralSpecies(
            # n_species=len(all_species),
            all_species=all_species,
            max_radial=hypers_ps["max_radial"],
            n_combined_basis=parameters.get("n_combined_basis", 16),
            n_pseudo_central_species=parameters.get("n_pseudo_central_species", len(all_species))
                if parameters.get("n_pseudo_central_species", len(all_species)) <= len(all_species)
                else len(all_species),
            seed=parameters.get("seed"),
            add_bias=parameters.get("combiner_add_bias", False),
            max_angular=hypers_ps["max_angular"]
        )
    elif COMBINER_TYPE == "CombineSpeciesWithCentralSpecies":
        assert species_center_key_to_samples == False
        combiner = CombineSpeciesWithCentralSpecies(
            all_species,
            hypers_ps["max_radial"],
            parameters["n_pseudo_species"],
            n_pseudo_central_species=parameters.get("n_pseudo_central_species", len(all_species))
                if parameters.get("n_pseudo_central_species", len(all_species)) <= len(all_species)
                else len(all_species),
            seed=parameters.get("seed")
        )

    COMPOSITION_REGULARIZER = parameters.get("composition_regularizer")
    RADIAL_SPECTRUM_REGULARIZER = parameters.get("radial_spectrum_regularizer")
    POWER_SPECTRUM_REGULARIZER = parameters.get("power_spectrum_regularizer")
    NN_REGULARIZER = parameters.get("nn_regularizer")
    FORCES_LOSS_WEIGHT = parameters.get("forces_loss_weight")
    OPTIMIZER_TYPE = parameters.get("optimizer", "LBFGS")
    PS_COMBINER_REGULARIZER = parameters.get("power_spectrum_combiner_regularizer", 0.0)
    STOP_EPOCH_SIZE = parameters.get("stop_epoch_size", 300)

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
        ) in train_dataloader_single_frame:
            model.initialize_model_weights(
                composition,
                radial_spectrum,
                spherical_expansions,
                energies,
                forces,
                seed=parameters.get("seed"),
            )
            break

    # crutch to manually move 'nn_model' to device
    # model.nn_model.to(device=device, dtype=torch.get_default_dtype()) # commented out

    del radial_spectrum, spherical_expansions
    n_parameters = sum(
        len(p.detach().cpu().numpy().flatten()) for p in model.parameters()
    )
    print(f"Model parameters: {n_parameters}")
    for name, param in model.named_parameters():
        print(f"{name}, {param.detach().cpu().numpy().shape}")

    now = datetime.now().strftime("%y%m%d-%H%M")
    prefix = f"{parameters['prefix']}_{now}"

    try:
        os.mkdir(prefix)
    except FileExistsError:
        pass

    json.dump(parameters, open(f"{prefix}/parameters.json", "w"))

    if parameters.get("restart", False):
        try:
            file_name = parameters.get("restart_file_name", "restart.torch")
            state = torch.load("./" + file_name)
            print("Restarting model parameters from file: " + file_name)
            model.load_state_dict(state)
        except FileNotFoundError:
            print("Restart file not found")

    optimizer = None
    if OPTIMIZER_TYPE == "LBFGS":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=parameters.get("learning_rate", 0.05),
            line_search_fn="strong_wolfe",
            history_size=128,
        )
    elif OPTIMIZER_TYPE == "ADAMW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parameters.get("learning_rate", 0.05),
        )
    elif OPTIMIZER_TYPE == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=parameters.get("learning_rate", 0.05),
        )

    output = open(f"{prefix}/log.txt", "w")
    if parameters.get("scheduler", False):
        output.write("# epoch  train_loss  test_mae test_mae_f  curr_lr\n")
    else:
        output.write("# epoch  train_loss  test_mae test_mae_f  epoch_plus_test_time  total_time\n")
    torch.save(model.state_dict(), f"{prefix}/initial.torch")

    high_mem = parameters.get("high_mem", True) # NOTE: Configuration "high_mem = False, n_train_forces != 0" needs to be tested
    if high_mem:
        composition, radial_spectrum, spherical_expansions, energies, forces = next(
            iter(train_dataloader_no_batch)
        )
        del train_dataset

        if n_train_forces != 0:
            (
                f_composition,
                f_radial_spectrum,
                f_spherical_expansions,
                f_energies,
                f_forces,
            ) = next(iter(train_forces_dataloader_grad_no_batch))
            del train_forces_dataset_grad

    best_mae = 1e100
    best_mae_epoch = None

    n_epochs_already_done = parameters.get("n_epochs_already_done", 0)
    n_epochs = parameters["n_epochs"]
    with open(f"{prefix}/epochs.dat", "a") as fd:
                fd.write("epoch test_mae test_mae_forces loss train_mae\n")

    scheduler = None
    if parameters.get("scheduler", False):
        lambda1 = lambda x: (1.0 + np.cos(10*(1+x)/np.pi/(n_epochs - n_epochs_already_done)))/2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    total_time = 0.0
    for epoch in range(n_epochs_already_done, n_epochs):
        print("Beginning epoch", epoch)
        epoch_start = time.time()

        @profile
        def single_step():
            # global composition, radial_spectrum, spherical_expansions, energies
            # global f_composition, f_radial_spectrum, f_spherical_expansions, f_energies, f_forces
            nonlocal composition, radial_spectrum, spherical_expansions, energies
            nonlocal f_composition, f_radial_spectrum, f_spherical_expansions, f_energies, f_forces, do_gradients
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
                if n_train_forces != 0:
                    f_predicted_e, f_predicted_f = model(
                        f_composition,
                        f_radial_spectrum,
                        f_spherical_expansions,
                        forward_forces=True,
                    )
                    loss += loss_mse(f_predicted_e, f_energies)
                    loss_force += loss_mse(f_predicted_f, f_forces)
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
                if n_train_forces != 0:
                    for (
                        f_composition,
                        f_radial_spectrum,
                        f_spherical_expansions,
                        f_energies,
                        f_forces,
                    ) in train_forces_dataloader_grad:
                        f_predicted_e, f_predicted_f = model(
                            f_composition,
                            f_radial_spectrum,
                            f_spherical_expansions,
                            forward_forces=True,
                        )
                        loss += loss_mse(f_predicted_e, f_energies)
                        loss_force += loss_mse(f_predicted_f, f_forces)

            loss /= n_train + n_train_forces
            loss_force /= n_train_forces

            if model.composition_model is not None:
                loss += COMPOSITION_REGULARIZER * torch.linalg.norm(
                    model.composition_model.weights
                )

            if model.radial_spectrum_model is not None:
                loss += RADIAL_SPECTRUM_REGULARIZER * torch.linalg.norm(
                    model.radial_spectrum_model.weights
                )

            if model.power_spectrum_model is not None:
                loss += POWER_SPECTRUM_REGULARIZER * torch.linalg.norm(
                    model.power_spectrum_model.weights
                )
                if PS_COMBINER_REGULARIZER != 0.0 and model.power_spectrum.combiner is not None:
                    loss += PS_COMBINER_REGULARIZER * sum(
                        (p**2).sum() for p in model.power_spectrum.combiner.parameters()
                    )

            if model.nn_model is not None:
                loss += NN_REGULARIZER * sum(
                    (p**2).sum() for p in model.nn_model.parameters()
                )

            energy_loss = loss.item()

            if n_train_forces != 0:
                loss += FORCES_LOSS_WEIGHT * loss_force

            print(
                f"    train loss: {loss.item()} E={energy_loss}, F={loss_force.item()}"
            )
            
            loss.backward(retain_graph=False)
            return loss

        if OPTIMIZER_TYPE == "LBFGS":
            loss = optimizer.step(single_step)
        else:
            for sub_epoch in range(25):
                loss = optimizer.step(single_step)

        epoch_time = time.time() - epoch_start
        if epoch % 1 == 0:
            train_mae = 0
            if high_mem:
                predicted, _ = model(
                            composition,
                            radial_spectrum,
                            spherical_expansions,
                            forward_forces=False,
                        )
                train_mae = loss_mae(predicted, energies)
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
                    train_mae+= loss_mae(predicted, energies)
            train_mae /= n_train + n_train_forces
            
            predicted = []
            reference = []
            predicted_forces = []
            reference_forces = []
            for (
                test_composition,
                test_radial_spectrum,
                test_spherical_expansions,
                test_energies,
                test_forces,
            ) in test_dataloader:
                reference.append(test_energies)
                test_predicted_e, test_predicted_f = model(
                    test_composition,
                    test_radial_spectrum,
                    test_spherical_expansions,
                    forward_forces=do_gradients,
                )
                predicted.append(test_predicted_e.detach())
                if do_gradients:
                    predicted_forces.append(test_predicted_f.detach())
                    reference_forces.append(test_forces)

            reference = torch.vstack(reference)
            predicted = torch.vstack(predicted)
            test_mae = loss_mae(predicted, reference) / n_test

            test_mae_forces = None
            if do_gradients:
                reference_forces = torch.vstack(reference_forces)
                predicted_forces = torch.vstack(predicted_forces)
                # TODO: do this properly
                test_mae_forces = (
                    loss_mae(predicted_forces, reference_forces) / n_test / (3 * 42)
                )
            else:
                class C(): # a class which returns a dummy value by calling ".item()"
                    item = lambda x: -111.111
                test_mae_forces = C()

            if scheduler is not None:
                curr_lr = optimizer.param_groups[0]["lr"]
                output.write(
                    f"{epoch} {loss.item()} {test_mae.item()} {test_mae_forces.item()} {curr_lr}\n"
                )
            else:
                epoch_plus_test_time = time.time() - epoch_start
                total_time+= epoch_plus_test_time
                str_total_time = strftime("%d-%H:%M:%S", gmtime(int(total_time)))
                # reduce the number of days by one
                str_total_time = f"{(int(str_total_time[:2]) - 1):02d}" + str_total_time[2:]
                output.write(
                    f"{epoch} {loss.item()} {test_mae.item()} {test_mae_forces.item()} {epoch_plus_test_time:.4} {str_total_time}\n"
                )
            output.flush()

            print(
                f"epoch {epoch} took {epoch_time:.4}s, "
                f"optimizer loss={loss.item():.4}, test mae={test_mae.item():.4} "
                f"test mae force={test_mae_forces.item():.4}"
            )

            with open(f"{prefix}/epochs.dat", "a") as fd:
                fd.write(f"{epoch} ")
                fd.write(f"{test_mae.item()} ")
                fd.write(f"{test_mae_forces.item()} ")
                fd.write(f"{loss.item()} ")
                fd.write(f"{train_mae.item()}\n")

            np.savetxt(
                f"{prefix}/energy_test.dat",
                np.hstack(
                    [reference.cpu().detach().numpy(), predicted.cpu().detach().numpy()]
                ),
            )
            if do_gradients:
                np.savetxt(
                    f"{prefix}/force_test.dat",
                    np.hstack(
                        [
                            reference_forces.cpu().detach().numpy().reshape(-1, 1),
                            predicted_forces.cpu().detach().numpy().reshape(-1, 1),
                        ]
                    ),
                )
            if test_mae < best_mae:
                best_mae = test_mae.detach().item()
                torch.save(model.state_dict(), f"{prefix}/best.torch")
                best_mae_epoch = epoch
            del test_mae, test_mae_forces

        if scheduler is not None:
            scheduler.step()

        del loss
        torch.save(model.state_dict(), f"{prefix}/restart.torch")
        
        if best_mae_epoch is not None and (epoch - best_mae_epoch >= STOP_EPOCH_SIZE):
            break

    torch.save(model.state_dict(), f"{prefix}/final.torch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"""
        This tool fits a potential for a multi-component system using an
        alchemical mixture model.

        Usage:
             python {sys.argv[0]} datafile.xyz parameters.json [-d device]
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
        "--device",
        type=str,
        default="cpu",
        help="torch device to run the model on",
    )
    args = parser.parse_args()

    main(args.datafile, args.parameters, args.device)
