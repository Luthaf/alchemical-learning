import argparse
import json
import os
import sys
import time
from datetime import datetime

import ase.io
import numpy as np
import torch

from utils.combine import UnitCombineSpecies, CombineSpecies
from utils.dataset import AtomisticDataset, create_dataloader
from utils.model import AlchemicalModel, SoapBpnn

try:
    profile
except NameError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func




torch.set_default_dtype(torch.float64)

#this is not the MAE
def loss_mae(predicted, actual):
    return torch.sum(torch.abs(predicted.flatten() - actual.flatten()))



def loss_mse(predicted, actual):
    return torch.mean((predicted.flatten() - actual.flatten()) ** 2)


def loss_rmse(predicted, actual):
    return torch.sqrt(loss_mse(predicted, actual))


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

    train_frames = frames[:n_train]
    train_forces_frames = frames[n_train : n_train + n_train_forces]
    test_frames = frames[-n_test:]

    train_energies, _ = extract_energy_forces(train_frames)
    train_f_energies, train_f_forces = extract_energy_forces(train_forces_frames)
    test_energies, test_forces = extract_energy_forces(test_frames)


    # determine the batch size of 
    batch_size_energy = parameters.get("batch_size", 32)
    ratio_forces_energies = n_train_forces / n_train
    batch_size_forces = batch_size_energy * ratio_forces_energies
    batch_size_forces = max(int(round(batch_size_forces, 0)),1) # at least batchsize 1 and integer

    print("batchsize energy: {}".format(batch_size_energy))
    print("batchsize forces: {}".format(batch_size_forces))

    print(
        f"using {n_train} training frames and {n_train_forces} training "
        "frames with forces"
    )

    
    print("Computing representations")
    hypers_ps = parameters["hypers_ps"]
    if "radial_per_angular" in hypers_ps:
        hypers_ps["radial_per_angular"] = {
            int(k): v for k, v in hypers_ps["radial_per_angular"].items()
        }
    hypers_rs = parameters.get("hypers_rs")

    # check that hypers_ps and hypers_rs have same cutoff radius

    assert hypers_rs["cutoff"] == hypers_ps["cutoff"]

    train_dataset = AtomisticDataset(
        train_frames,
        all_species,
        {"radial_spectrum": hypers_rs, "spherical_expansion": hypers_ps},
        train_energies,
    )

    if n_train_forces != 0:
        train_forces_dataset_grad = AtomisticDataset(
            train_forces_frames,
            all_species,
            {"radial_spectrum": hypers_rs, "spherical_expansion": hypers_ps},
            train_f_energies,
            train_f_forces,
            do_gradients=True,
        )
    else:
        train_forces_dataset_grad = None

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
    train_dataloader_no_batch = create_dataloader(
        train_dataset,
        batch_size=batch_size_energy,
        shuffle=True,
        device=device,
    )

    train_dataloader_single_frame = create_dataloader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        device=device,
    )

    if n_train_forces != 0:
        #How do we handle grads in SGD?
        # One force batch per energy batch ? frac_ E_samples/ 
        # Actually we need one energy and one force dataloader
        # for every energy batch, we can call one force batch

        train_forces_dataloader_grad_no_batch = create_dataloader(
            train_forces_dataset_grad,
            batch_size=batch_size_forces,
            shuffle=True,
            device=device,
        )

    else:
        train_forces_dataloader_grad_no_batch = None
    
    n_batches_energy = len(train_dataloader_no_batch)
    n_batches_forces = len(train_forces_dataloader_grad_no_batch)

    print("N batches train energies: {}".format(n_batches_energy))
    print("N batches train forces: {}".format(n_batches_forces))

    #Why? I do not understand
    test_dataloader_grad = create_dataloader(
        test_dataset_grad,
        batch_size=n_test,
        shuffle=False,
        device=device,
    )

    combiner = UnitCombineSpecies(
        species=all_species,
        n_pseudo_species=parameters["n_pseudo_species"],
        # TODO: remove this from code
        per_l_max=0,
    )

    COMPOSITION_REGULARIZER = parameters.get("composition_regularizer")
    NN_REGULARIZER = parameters.get("nn_regularizer")
    FORCES_LOSS_WEIGHT = parameters.get("forces_loss_weight")

    model = SoapBpnn(
        combiner=combiner,
        composition_regularizer=COMPOSITION_REGULARIZER,
        nn_layer_size=parameters.get("nn_layer_size"),
        ps_center_types=None,
    )

    model.to(device=device, dtype=torch.get_default_dtype())

    # --------- INITIALIZE MODEL --------- #
    print("Initializing model")
    print("Setting seed to: {}".format(parameters.get("seed")))
    torch.manual_seed(int(parameters.get("seed")))


    #TODO: abstract this in some model.initialize() method
    
    with torch.no_grad():
        for (
            composition,
            radial_spectrum,
            spherical_expansions,
            energies,
            forces,
        ) in train_dataloader_single_frame:

            class CustomMLP(torch.nn.Module):
                def __init__(self, dim_input,layer_size,dim_output):
                    super().__init__()

                    self.dim_input = dim_input
                    self.layer_size = layer_size
                    self.dim_output = dim_output

                    self.nn =   torch.nn.Sequential(
                                torch.nn.Linear(self.dim_input, self.layer_size),
                                torch.nn.Tanh(),
                                torch.nn.LayerNorm(self.layer_size),
                                torch.nn.Linear(self.layer_size, self.layer_size),
                                torch.nn.Tanh(),
                                torch.nn.LayerNorm(self.layer_size),
                                torch.nn.Linear(self.layer_size, self.dim_output),
                                )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.nn(x)

            model.initialize_model_weights(
                composition,
                radial_spectrum,
                spherical_expansions,
                energies,
                forces,
                seed=parameters.get("seed"),
                nn_architecture=CustomMLP
            )
            break

    del radial_spectrum, spherical_expansions

    # TODO: abstract this in some .get_model_info() method
    n_parameters = sum(
        len(p.detach().cpu().numpy().flatten()) for p in model.parameters()
    )
    print(f"Model parameters: {n_parameters}")


    now = datetime.now().strftime("%y%m%d-%H%M")
    prefix = f"{parameters['prefix']}_{now}"

    try:
        os.mkdir(prefix)
    except FileExistsError:
        pass



    #This could be part of a Trainer module
    json.dump(parameters, open(f"{prefix}/parameters.json", "w"))


    if parameters.get("restart", False):
        try:
            state = torch.load("./restart.torch")
            print("Restarting model parameters from file")
            model.load_state_dict(state)
        except FileNotFoundError:
            print("Restart file not found")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters.get("learning_rate", 0.001),
    )


    # saves initial model dict
    output = open(f"{prefix}/log.txt", "w")
    output.write("# epoch  train_loss  test_mae test_mae_f\n")
    torch.save(model.state_dict(), f"{prefix}/initial.torch")

    high_mem = True
    if high_mem:
        
        composition, radial_spectrum, spherical_expansions, energies, forces = next(
            iter(train_dataloader_no_batch)
        )
        del train_dataset

        #why?

        """
        if n_train_forces == 0:
            pass
            
            (
                f_composition,
                f_radial_spectrum,
                f_spherical_expansions,
                f_energies,
                f_forces,
            ) = next(iter(train_forces_dataloader_grad_no_batch))
        
        else:
            assert False
        """

        # No
        #del train_forces_dataset_grad
    
    best_mae = 1e100

    n_epochs_already_done = parameters.get("n_epochs_already_done", 0)
    n_epochs = parameters["n_epochs"]
    with open(f"{prefix}/epochs.dat", "a") as fd:
                fd.write("epoch test_mae test_mae_forces loss train_mae\n")
    
                
    for epoch in range(n_epochs_already_done, n_epochs):
        print("Beginning epoch", epoch)
        epoch_start = time.time() 

        n_batches_in_energy_loader = len(train_dataloader_no_batch)
        n_batches_in_force_loader = len(train_forces_dataloader_grad_no_batch)
        assert n_batches_in_energy_loader == n_batches_in_force_loader 

        for (
            composition,
            radial_spectrum,
            spherical_expansions,
            energies,
            forces,
        ),  (
                f_composition,
                f_radial_spectrum,
                f_spherical_expansions,
                f_energies,
                f_forces,
            ) in zip(train_dataloader_no_batch,train_forces_dataloader_grad_no_batch):


            optimizer.zero_grad()
            loss = torch.zeros(size=(1,), device=device)
            loss_force = torch.zeros(size=(1,), device=device)
            assert high_mem

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

            if model.composition_model is not None:
                loss += COMPOSITION_REGULARIZER * torch.linalg.norm(
                    model.composition_model.weights
                )

            if model.nn_model is not None:
                loss += NN_REGULARIZER * sum(
                    (p**2).sum() for p in model.nn_model.parameters()
                )

            energy_loss = loss.item()

            if n_train_forces != 0:
                loss += FORCES_LOSS_WEIGHT * loss_force


            
            loss.backward(retain_graph=False)
            optimizer.step()

        loss /= n_train + n_train_forces
        loss_force /= n_train_forces

        print(
            f"    train loss: {loss.item()} E={energy_loss}, F={loss_force.item()}"
        )

        epoch_time = time.time() - epoch_start
        
        if epoch % 1 == 0:
            
            #TODO: This has to be removed or changed in Batch SGD
            predicted, _ = model(
                        composition,
                        radial_spectrum,
                        spherical_expansions,
                        forward_forces=False,
                    )   
            train_mae = loss_mae(predicted, energies)                   
            train_mae /= n_train + n_train_forces
            
            predicted = []
            reference = []
            predicted_forces = []
            reference_forces = []
            
            # we can keep a full batch testing loader
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

            reference = torch.vstack(reference)
            predicted = torch.vstack(predicted)

            test_mae = loss_mae(predicted, reference) / n_test
            test_rmse = loss_rmse(predicted, reference)

            reference_forces = torch.vstack(reference_forces)
            predicted_forces = torch.vstack(predicted_forces)
            # TODO: do this properly
            
            test_mae_forces = (
                loss_mae(predicted_forces, reference_forces) / n_test / (3 * 64 * 3)
            )
            
            test_rmse_forces = (
                loss_rmse(predicted_forces.flatten(), reference_forces.flatten())
            )


            output.write(
                f"{epoch} {loss.item()} {test_mae.item()} {test_mae_forces.item()}\n"
            )
            output.flush()

            print(
                f"epoch {epoch} took {epoch_time:.4}s, "
                f"optimizer loss={loss.item():.4}, test mae={test_mae.item():.4}, test rmse={test_rmse.item():.4}, "
                f"test mae force={test_mae_forces.item():.4}, test rmse force={test_rmse_forces.item():.4},"
            )

            with open(f"{prefix}/epochs.dat", "a") as fd:
                fd.write(f"{epoch} ")
                fd.write(f"{test_mae.item()} ")
                fd.write(f"{test_rmse.item()} ")
                fd.write(f"{test_mae_forces.item()} ")
                fd.write(f"{loss.item()} ")
                fd.write(f"{train_mae.item()}\n")

            np.savetxt(
                f"{prefix}/energy_test.dat",
                np.hstack(
                    [reference.cpu().detach().numpy(), predicted.cpu().detach().numpy()]
                ),
            )


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
            del test_mae, test_mae_forces

        del loss
        torch.save(model.state_dict(), f"{prefix}/restart.torch")

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
