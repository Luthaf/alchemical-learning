#command-line version of the HEA-multibody notebook (with force learning)

import copy
import cProfile
import time

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import torch

from utils.combine import CombineRadial, CombineRadialSpecies, CombineSpecies
from utils.dataset import AtomisticDataset, create_dataloader
from utils.linear import LinearModel
from utils.operations import SumStructures, remove_gradient
from utils.soap import PowerSpectrum, CompositionFeatures

torch.set_default_dtype(torch.float64)

n_test = 10
n_train = 80
n_train_forces = 20
prefix = "sigma0.25"

frames = ase.io.read("data/data_shuffle.xyz", f":{n_test + n_train + n_train_forces}")

train_frames = frames[:n_train]
train_forces_frames = frames[n_train:n_train+n_train_forces]
test_frames = frames[-n_test:]

train_energies = torch.tensor(
    [frame.info["energy"] for frame in train_frames]
).reshape(-1, 1).to(dtype=torch.get_default_dtype())

test_energies = torch.tensor(
    [frame.info["energy"] for frame in test_frames]
).reshape(-1, 1).to(dtype=torch.get_default_dtype())


train_forces_e = torch.tensor(
    [frame.info["energy"] for frame in train_forces_frames]
).reshape(-1, 1).to(dtype=torch.get_default_dtype())

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

# HYPERS_FROM_PAPER = {
#     "interaction_cutoff": 5.0,
#     "max_angular": 9,
#     "max_radial": 12,
#     "gaussian_sigma_constant": 0.3,
#     "gaussian_sigma_type": "Constant",
#     "cutoff_smooth_width": 0.5,
#     "radial_basis": "GTO",
#     "compute_gradients": False,
#     "expansion_by_species_method": "user defined",
#     "global_species": all_species,
# }

HYPERS_SMALL = {
    "cutoff": 4.0,
    "max_angular": 3,
    "max_radial": 5,
    "atomic_gaussian_width": 0.3,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"SplinedGto": {"accuracy": 1e-6}},
    "gradients": False,
    "center_atom_weight": 1.0,
    "radial_scaling":  {"Willatt2018": { "scale": 2.5, "rate": 0.8, "exponent": 2}}
    # # TODO: implement this in rascaline itself
    # "radial_per_angular": {
    #     # l: n
    #     0: 10,
    #     1: 8,
    #     2: 8,
    #     3: 4,
    #     4: 4,
    # }
}

HYPERS_RADIAL = {
    "cutoff": 6.0,
    "max_angular": 0,
    "max_radial": 12,
    "atomic_gaussian_width": 0.3,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"SplinedGto": {"accuracy": 1e-6}},
    "gradients": False,
    "center_atom_weight": 1.0,    
    "radial_scaling":  {"Willatt2018": { "scale": 2.5, "rate": 0.8, "exponent": 2}}
}


device = "cpu"

#if torch.cuda.is_available():
#    device = "cuda"

train_dataset = AtomisticDataset(train_frames, all_species, 
                                 {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion":HYPERS_SMALL}, train_energies)
train_forces_dataset = AtomisticDataset(train_forces_frames, all_species, 
                                 {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion":HYPERS_SMALL}, train_forces_e)
test_dataset = AtomisticDataset(test_frames, all_species, 
                                {"radial_spectrum": HYPERS_RADIAL, "spherical_expansion":HYPERS_SMALL}, test_energies)                                
do_gradients = True
if do_gradients is True:
    HYPERS_GRAD = copy.deepcopy(HYPERS_SMALL)
    HYPERS_GRAD["gradients"] = do_gradients
    HYPERS_RAD_GRAD = copy.deepcopy(HYPERS_RADIAL)
    HYPERS_RAD_GRAD["gradients"] = do_gradients
    train_forces_dataset_grad = AtomisticDataset(train_forces_frames, all_species, 
                                          {"radial_spectrum": HYPERS_RAD_GRAD, "spherical_expansion":HYPERS_GRAD}, 
                                          train_forces_e, train_forces_f)
    test_dataset_grad = AtomisticDataset(test_frames, all_species, 
                                         {"radial_spectrum": HYPERS_RAD_GRAD, "spherical_expansion":HYPERS_GRAD}, 
                                         test_energies, test_forces)
else:
    train_forces_dataset_grad = train_forces_dataset
    test_dataset_grad = test_dataset
    
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
    batch_size=len(train_dataset),
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
    batch_size=200,
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
        shuffle=True,
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
    return torch.sum((predicted.flatten() - actual.flatten())**2)

def loss_rmse(predicted, actual):
    return np.sqrt(loss_mse(predicted, actual))

class CombinedPowerSpectrum(torch.nn.Module):
    def __init__(self, combiner):
        super().__init__()

        self.combiner = combiner
        self.power_spectrum = PowerSpectrum()

    def forward(self, spherical_expansion):
        combined = self.combiner(spherical_expansion)

        return self.power_spectrum(combined)

        
class MultiBodyOrderModel(torch.nn.Module):
    def __init__(
        self, 
        power_spectrum,
        composition_regularizer,
        radial_spectrum_regularizer,
        power_spectrum_regularizer,        
        optimizable_weights,
        random_initial_weights,
    ):
        super().__init__()

        self.sum_structure = SumStructures()

        # optimizable_weights = False is not very well tested ...
        assert optimizable_weights

        if composition_regularizer is None:
            self.composition_model = None
        else:
            self.composition_model=LinearModel(
            regularizer=composition_regularizer,
            optimizable_weights=optimizable_weights,
            random_initial_weights=random_initial_weights,
        )
        
        if radial_spectrum_regularizer is None:
            self.radial_spectrum_model = None
        else:
            self.radial_spectrum_model = LinearModel(
                regularizer=radial_spectrum_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        if power_spectrum_regularizer is None:
            self.power_spectrum_model = None
        else:
            self.power_spectrum = power_spectrum
            self.power_spectrum_model = LinearModel(
                regularizer=power_spectrum_regularizer,
                optimizable_weights=optimizable_weights,
                random_initial_weights=random_initial_weights,
            )

        self.combiner = combiner, 
        self.optimizable_weights = optimizable_weights
        self.random_initial_weights = random_initial_weights

    def forward(self, composition, radial_spectrum, spherical_expansion, forward_forces=False):
        if not forward_forces:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)                
    
        energies, forces = None, None
        
        if self.composition_model is not None:
            energies_cmp, _ = self.composition_model(composition)
            energies = energies_cmp
            forces = None
    
        if self.radial_spectrum_model is not None:
            radial_spectrum_per_structure = radial_spectrum #self.sum_structure(radial_spectrum)
            energies_rs, forces_rs = self.radial_spectrum_model(radial_spectrum_per_structure, with_forces=forward_forces)
            
            if energies is None:
                energies = energies_rs  
            else:
                energies += energies_rs              
            if forces_rs is not None:
                if forces is None:
                    forces = forces_rs
                else:
                    forces += forces_rs

        if self.power_spectrum_model is not None:
            power_spectrum = self.power_spectrum(spherical_expansion)
            power_spectrum_per_structure = self.sum_structure(power_spectrum)

            energies_ps, forces_ps = self.power_spectrum_model(power_spectrum_per_structure, with_forces=forward_forces)
            if energies is None:
                energies = energies_ps
            else:
                energies += energies_ps
            if forces_ps is not None:
                if forces is None:
                    forces = forces_ps
                else:
                    forces += forces_ps
        
        return energies, forces

    def initialize_model_weights(self, composition, radial_spectrum, spherical_expansion, energies, forces=None, seed=None):
        if forces is None:
            # remove gradients if we don't need them
            spherical_expansion = remove_gradient(spherical_expansion)
            if radial_spectrum is not None:
                radial_spectrum = remove_gradient(radial_spectrum)
            
        if self.composition_model is not None:
            self.composition_model.initialize_model_weights(composition, energies, forces, seed)
        
        if self.radial_spectrum_model is not None:
            radial_spectrum_per_structure = self.sum_structure(radial_spectrum)
            self.radial_spectrum_model.initialize_model_weights(radial_spectrum_per_structure, energies, forces, seed)
        
        if self.power_spectrum_model is not None:        
            power_spectrum = self.power_spectrum(spherical_expansion)
            power_spectrum_per_structure = self.sum_structure(power_spectrum)
            self.power_spectrum_model.initialize_model_weights(power_spectrum_per_structure, energies, forces, seed)

# species combination only
N_PSEUDO_SPECIES = 4
combiner = CombineSpecies(species=all_species, n_pseudo_species=N_PSEUDO_SPECIES)

# # species combination and then radial basis combination
# N_COMBINED_RADIAL = 4
# combiner = torch.nn.Sequential(
#     CombineSpecies(species=all_species, n_pseudo_species=N_PSEUDO_SPECIES),
#     CombineRadial(max_radial=HYPERS_SMALL["max_radial"], n_combined_radial=N_COMBINED_RADIAL),
# )

# # combine both radial and species information at the same time
# combiner = CombineRadialSpecies(
#     n_species=len(all_species), 
#     max_radial=HYPERS_SMALL["max_radial"], 
#     n_combined_basis=N_COMBINED_RADIAL*N_PSEUDO_SPECIES,
# )

composition=CompositionFeatures(all_species, device=device)
power_spectrum = CombinedPowerSpectrum(combiner)

LINALG_REGULARIZER_ENERGIES = 1e-2
LINALG_REGULARIZER_FORCES = 1e-1

model = MultiBodyOrderModel(
    power_spectrum=power_spectrum, 
    composition_regularizer=[1e-10],
    radial_spectrum_regularizer=[LINALG_REGULARIZER_ENERGIES, LINALG_REGULARIZER_FORCES],
    power_spectrum_regularizer=[LINALG_REGULARIZER_ENERGIES, LINALG_REGULARIZER_FORCES],
    optimizable_weights=True, 
    random_initial_weights=True,
)

if model.optimizable_weights:
    TORCH_REGULARIZER_COMPOSITION = 1e-9
    TORCH_REGULARIZER_RADIAL_SPECTRUM = 1e-4
    TORCH_REGULARIZER_POWER_SPECTRUM = 1e-4
else:
    TORCH_REGULARIZER_RADIAL_SPECTRUM = 0.0
    TORCH_REGULARIZER_POWER_SPECTRUM = 0.0

    
model.to(device=device, dtype=torch.get_default_dtype())

if model.random_initial_weights:
    dataloader_initialization = train_forces_dataloader_grad_single_frame
else:
    dataloader_initialization = train_dataloader_no_batch
    
    
# initialize the model
with torch.no_grad():
    for composition, radial_spectrum, spherical_expansions, energies, forces in dataloader_initialization:
        # we want to intially train the model on all frames, to ensure the
        # support points come from the full dataset.
        model.initialize_model_weights(composition, radial_spectrum, spherical_expansions, energies, forces, seed=12345)
        break

del radial_spectrum, spherical_expansions

lr = 0.1
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)#, line_search_fn="strong_wolfe", history_size=128)

all_losses = []
all_tests=[]
f_all_tests=[]

filename = f"{prefix}-{model.__class__.__name__}-{N_PSEUDO_SPECIES}-mixed-{n_train}-f{n_train_forces}-train"
if model.optimizable_weights:
    filename += "-opt-weights"

if model.random_initial_weights:
    filename += "-random-weights"

output = open(f"{filename}.dat", "w")
output.write("# epoch  train_loss  test_mae test_mae_f\n")
n_epochs_total = 0

torch.save(model.state_dict(), f"{filename}-init.torch")

assert model.optimizable_weights
himem = True
if himem and len(all_losses)==0:
    composition, radial_spectrum, spherical_expansions, energies, forces = next(iter(train_dataloader_no_batch))
    f_composition, f_radial_spectrum, f_spherical_expansions, f_energies, f_forces = next(iter(train_forces_dataloader_grad_no_batch))

for epoch in range(10):
    epoch_start = time.time()

    def single_step():
        global composition, radial_spectrum, spherical_expansions, energies
        optimizer.zero_grad()
        if device=="cuda":
            print(f"mem. before:  {torch.cuda.memory_stats()['allocated_bytes.all.current']/1e6} MB allocated, {torch.cuda.memory_stats()['reserved_bytes.all.current']/1e6} MB reserved ")
        loss = torch.zeros(size=(1,), device=device)
        loss_force = torch.zeros(size=(1,), device=device)
        if himem:
            predicted, _ = model(composition, radial_spectrum, spherical_expansions, forward_forces=False)
            loss += loss_mse(predicted, energies)
            f_predicted_e, f_predicted_f = model(f_composition, f_radial_spectrum, f_spherical_expansions, forward_forces=do_gradients)
            loss += loss_mse(f_predicted_e, f_energies)
            if do_gradients:
                loss_force += loss_mse(f_predicted_f, f_forces)/42
        else:
            for composition, radial_spectrum, spherical_expansions, energies, forces in train_dataloader:
                try:
                    predicted, _ = model(composition, radial_spectrum, spherical_expansions, forward_forces=False)
                except:
                    if device=="cuda":
                        print(f"mem. during:  {torch.cuda.memory_stats()['allocated_bytes.all.current']/1e6} MB allocated, {torch.cuda.memory_stats()['reserved_bytes.all.current']/1e6} MB reserved ")
                    raise
                loss += loss_mse(predicted, energies)
            raise ValueError("MUST IMPLEMENT FORCE CALCULATOR FOR THIS PATH!")
        loss /= n_train
        if model.composition_model is not None:
            loss += TORCH_REGULARIZER_COMPOSITION * torch.linalg.norm(model.composition_model.weights)
        if model.radial_spectrum_model is not None:
            loss += TORCH_REGULARIZER_RADIAL_SPECTRUM * torch.linalg.norm(model.radial_spectrum_model.weights)
        if model.power_spectrum_model is not None:
            loss += TORCH_REGULARIZER_POWER_SPECTRUM * torch.linalg.norm(model.power_spectrum_model.weights)

        print(f"Train loss: {(loss+loss_force).item()} E={loss.item()}, F={loss_force.item()}")
        loss+=loss_force
        loss.backward(retain_graph=False)
        print("Loss gradient", np.linalg.norm(model.composition_model.weights.grad.numpy()))
        return loss
            
    loss = optimizer.step(single_step)
    loss = loss.item()
    all_losses.append(loss)

    epoch_time = time.time() - epoch_start
    if epoch % 1 == 0:
        print("norms", np.linalg.norm(0 if model.composition_model is None else model.composition_model.weights.detach().cpu().numpy()),
                  np.linalg.norm(0 if model.radial_spectrum_model is None else model.radial_spectrum_model.weights.detach().cpu().numpy()),
                  np.linalg.norm(0 if model.power_spectrum_model is None else model.power_spectrum_model.weights.detach().cpu().numpy())
                 )
        print("gradients", 
                  np.linalg.norm(0 if model.composition_model is None else model.composition_model.weights.grad.detach().cpu().numpy()),
                  np.linalg.norm(0 if model.radial_spectrum_model is None else model.radial_spectrum_model.weights.grad.detach().cpu().numpy()),
                  np.linalg.norm(0 if model.power_spectrum_model is None else model.power_spectrum_model.weights.grad.detach().cpu().numpy())
                 )
        with torch.no_grad():
            predicted = []
            reference = []
            f_predicted = []
            f_reference = []
            for tcomposition, tradial_spectrum, tspherical_expansions, tenergies, tforces in test_dataloader_grad:
                reference.append(tenergies)
                tpredicted_e, tpredicted_f = model(tcomposition, tradial_spectrum, tspherical_expansions, forward_forces=do_gradients)
                predicted.append(tpredicted_e)                
                if do_gradients:
                    f_predicted.append(tpredicted_f)
                    f_reference.append(tforces)

            reference = torch.vstack(reference)
            predicted = torch.vstack(predicted)
            test_mae = loss_mae(predicted, reference)/n_test
            output.write(f"{n_epochs_total} {loss} {test_mae}")
            if do_gradients:
                f_reference = torch.vstack(f_reference)
                f_predicted = torch.vstack(f_predicted)
                f_test_mae = loss_mae(f_predicted, f_reference)/n_test/42
                output.write(f"{f_test_mae}")                
            output.write("\n")
            output.flush()
        all_tests.append(test_mae.item())
        f_all_tests.append(f_test_mae.item())
        print(f"epoch {n_epochs_total} took {epoch_time:.4}s, optimizer loss={loss:.4}, test mae={test_mae:.4}"+
              (f" test mae force={f_test_mae:.4}"))
        np.savetxt(f"{filename}-energy_test.dat",np.hstack([reference.cpu().numpy(), predicted.cpu().numpy()]))
        np.savetxt(f"{filename}-force_test.dat",np.hstack([f_reference.cpu().numpy(), f_predicted.cpu().numpy()]))
    del loss
    n_epochs_total += 1
    torch.save(model.state_dict(), f"{filename}-restart.torch")

torch.save(model.state_dict(), f"{filename}-final.torch")


with torch.no_grad():
    tpredicted = []
    treference = []
    for tcomposition, tradial_spectrum, tspherical_expansions, tenergies, _ in train_dataloader:
        treference.append(tenergies)
        predicted_e, _ = model(tcomposition, tradial_spectrum, tspherical_expansions, forward_forces=False)
        tpredicted.append(predicted_e)

    treference = torch.vstack(treference)
    tpredicted = torch.vstack(tpredicted)
    tmae = loss_mae(tpredicted, treference)/n_train

print(f"TRAIN MAE: {tmae.item()/42} eV/at")    
print(f"TEST MAE: {test_mae.item()/42} eV/at")