import copy

import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap
from rascaline import SphericalExpansion
from .operations import SumStructures
from time import time
from .soap import CompositionFeatures

def _block_to_torch(block, structure_i):
    assert block.samples.names[0] == "structure"    
    samples = block.samples.view(dtype=np.int32).reshape(-1, len(block.samples.names)).copy()
    samples[:, 0] = structure_i
    samples = Labels(block.samples.names, samples)

    new_block = TensorBlock(
        values=torch.tensor(block.values).to(dtype=torch.get_default_dtype()),
        samples=samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter in block.gradients_list():
        gradient = block.gradient(parameter)

        assert gradient.samples.names == ("sample", "structure", "atom")

        gradient_samples = gradient.samples.view(dtype=np.int32).reshape(-1, len(gradient.samples.names)).copy()
        gradient_samples[:, 1] = structure_i
        gradient_samples = Labels(gradient.samples.names, gradient_samples)

        new_block.add_gradient(
            parameter=parameter,
            data=torch.tensor(gradient.data).to(dtype=torch.get_default_dtype()),
            samples=gradient_samples,
            components=gradient.components,
        )

    return new_block


def _move_to_torch(tensor_map, structure_i):
    blocks = []
    for _, block in tensor_map:
        blocks.append(_block_to_torch(block, structure_i))

    return TensorMap(tensor_map.keys, blocks)


def _move_to_torch_by_l(tensor_maps, structure_i):
    keys = Labels(
        names=["spherical_harmonics_l"],
        values=np.array(list(tensor_maps.keys()), dtype=np.int32).reshape(-1, 1),
    )

    blocks = []
    for l, tensor_map in tensor_maps.items():
        block = tensor_map.block(spherical_harmonics_l=l)
        blocks.append(_block_to_torch(block, structure_i))

    return TensorMap(keys, blocks)


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames,
        all_species,
        hypers,
        energies,
        forces=None,
        radial_spectrum_n_max=None,
        radial_spectrum_rcut=None,
    ):
        all_neighbor_species = Labels(
            names=["species_neighbor"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )
        all_center_species = Labels(
            names=["species_center"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )       
            
        self.radial_spectrum = []
        hypers = copy.deepcopy(hypers)
        if radial_spectrum_n_max is not None:
            new_hypers = copy.deepcopy(hypers)
            new_hypers["max_angular"] = 0
            new_hypers["max_radial"] = radial_spectrum_n_max
            if radial_spectrum_rcut is not None:
                new_hypers["cutoff"] = radial_spectrum_rcut
            calculator = SphericalExpansion(**new_hypers)
            summer = SumStructures()

            for frame_i, frame in enumerate(frames):
                spherical_expansion = calculator.compute(frame)
                spherical_expansion.keys_to_properties(all_center_species)

                # TODO: these don't hurt, but are confusing, let's remove them
                spherical_expansion.components_to_properties("spherical_harmonics_m")
                spherical_expansion.keys_to_properties("spherical_harmonics_l")

                spherical_expansion.keys_to_properties(all_neighbor_species)                
                #sph_structure = summer(spherical_expansion)
                self.radial_spectrum.append(
                    summer(_move_to_torch(spherical_expansion, frame_i))
                )
        else:
            self.radial_spectrum = [None] * len(frames)

        self.spherical_expansions = []
        if "radial_per_angular" in hypers:
            radial_per_angular = hypers.pop("radial_per_angular")

            calculators = {}
            for l, n in radial_per_angular.items():
                new_hypers = copy.deepcopy(hypers)
                new_hypers["max_angular"] = l
                new_hypers["max_radial"] = n
                calculators[l] = SphericalExpansion(**new_hypers)

            for frame_i, frame in enumerate(frames):
                spherical_expansion_by_l = {}
                for l, calculator in calculators.items():
                    spherical_expansion = calculator.compute(frame)
                    spherical_expansion.keys_to_samples("species_center")
                    spherical_expansion.keys_to_properties(all_neighbor_species)
                    spherical_expansion_by_l[l] = spherical_expansion

                self.spherical_expansions.append(
                    _move_to_torch_by_l(spherical_expansion_by_l, frame_i)
                )

        else:
            calculator = SphericalExpansion(**hypers)

            for frame_i, frame in enumerate(frames):
                spherical_expansion = calculator.compute(frame)
                spherical_expansion.keys_to_samples("species_center")
                spherical_expansion.keys_to_properties(all_neighbor_species)

                self.spherical_expansions.append(
                    _move_to_torch(spherical_expansion, frame_i)
                )

        self.composition = []
        if all_species is not None:
            comp_calc=CompositionFeatures(all_species)   
            for frame_i, frame in enumerate(frames):
                comp=comp_calc([frame], np.array([frame_i], dtype=np.int32).reshape(-1,1))
                self.composition.append(
                    comp
                )
        else:
            self.composition = [None] * len(frames)                
                
        assert isinstance(energies, torch.Tensor)
        assert energies.shape == (len(frames), 1)
        self.energies = energies

        if forces is not None:
            assert isinstance(forces, list)
            for i, f in enumerate(forces):
                assert isinstance(f, torch.Tensor)
                assert f.shape == (len(self.frames[i]), 3)

        self.forces = forces
        self._getitemtime = 0
        self._collatetime = 0

    def __len__(self):
        return len(self.composition)

    def __getitem__(self, idx):
        self._getitemtime -= time()
        if self.forces is None:
            forces = None
        else:
            forces = self.forces[idx]        

        data = (
            self.composition[idx],
            self.radial_spectrum[idx],
            self.spherical_expansions[idx],
            self.energies[idx],
            forces,
        )
        self._getitemtime += time()
        return data


def _collate_tensor_map(tensors, device):
    keys = tensors[0].keys

    blocks = []
    for key in keys:
        first_block = tensors[0].block(key)
        if first_block.has_gradient("positions"):
            first_block_grad = first_block.gradient("positions")
        else:
            first_block_grad = None

        samples = []
        values = []

        grad_samples = []
        grad_data = []
        previous_samples_count = 0
        for tensor in tensors:
            block = tensor.block(key)

            new_samples = block.samples.view(dtype=np.int32).reshape(
                -1, len(block.samples.names)
            )
            samples.append(new_samples)
            values.append(block.values)

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")

                new_grad_samples = (
                    gradient.samples.view(dtype=np.int32).reshape(-1, 3).copy()
                )
                new_grad_samples[:, 0] += previous_samples_count
                grad_samples.append(new_grad_samples)

                grad_data.append(gradient.data)

            previous_samples_count += new_samples.shape[0]
        new_block = TensorBlock(
            values=torch.vstack(values).to(device),
            samples=Labels(
                block.samples.names,
                np.vstack(samples),
            ),
            components=first_block.components,
            properties=first_block.properties,
        )
        if first_block_grad is not None:
            new_block.add_gradient(
                "positions",
                data=torch.vstack(grad_data).to(device),
                samples=Labels(
                    ["sample", "structure", "atom"],
                    np.vstack(grad_samples),
                ),
                components=first_block_grad.components,
            )

        blocks.append(new_block)

    return TensorMap(keys, blocks)


def _collate_data(device, self):
    def do_collate(data):
        start=time()
        composition = [d[0] for d in data]
        if composition[0] is None:
            composition = None
        else:
            composition = _collate_tensor_map(composition, device)

        radial_spectrum = [d[1] for d in data]
        if radial_spectrum[0] is None:
            radial_spectrum = None
        else:
            radial_spectrum = _collate_tensor_map(radial_spectrum, device)

        spherical_expansion = _collate_tensor_map([d[2] for d in data], device)

        energies = torch.vstack([d[3] for d in data]).to(device=device)
        if data[0][4] is not None:
            forces = torch.vstack([d[4] for d in data]).to(device=device)
        else:
            forces = None
        self._collatetime+=time()-start
        return composition, radial_spectrum, spherical_expansion, energies, forces

    return do_collate


def create_dataloader(dataset, batch_size, shuffle=True, device="cpu"):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_data(device, dataset),
    )
