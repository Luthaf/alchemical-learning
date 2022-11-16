import copy
from time import time

import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap
from rascaline import SphericalExpansion
from rascaline_torch import Calculator, as_torch_system

from .operations import SumStructures
from .soap import CompositionFeatures


def _block_to_torch(block, structure_i):
    assert block.samples.names[0] == "structure"
    samples = (
        block.samples.view(dtype=np.int32).reshape(-1, len(block.samples.names)).copy()
    )
    samples[:, 0] = structure_i
    samples = Labels(block.samples.names, samples)

    new_block = TensorBlock(
        values=block.values,
        samples=samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter in block.gradients_list():
        gradient = block.gradient(parameter)

        gradient_samples = (
            gradient.samples.view(dtype=np.int32)
            .reshape(-1, len(gradient.samples.names))
            .copy()
        )
        if gradient.samples.names == ("sample", "structure", "atom"):
            gradient_samples[:, 1] = structure_i
        gradient_samples = Labels(gradient.samples.names, gradient_samples)

        new_block.add_gradient(
            parameter=parameter,
            data=gradient.data,
            samples=gradient_samples,
            components=gradient.components,
        )

    return new_block


def _move_to_torch(tensor_map, structure_i, detach=False):
    blocks = []
    for _, block in tensor_map:
        blocks.append(_block_to_torch(block, structure_i))

    return TensorMap(tensor_map.keys, blocks)


def _detach_all_blocks(tensor_map):
    blocks = []
    for _, block in tensor_map:
        new_block = TensorBlock(
            values=block.values.detach(),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)

            new_block.add_gradient(
                parameter=parameter,
                data=gradient.data.detach(),
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)
    return TensorMap(tensor_map.keys, blocks)


def _move_to_torch_by_l(tensor_maps, structure_i):
    keys = []
    blocks = []
    for l, tensor_map in tensor_maps.items():
        l_blocks = tensor_map.block(spherical_harmonics_l=l)
        if not hasattr(l_blocks, "__len__"):
            l_blocks = [l_blocks]
        l_keys = tensor_map.keys[
            np.where(tensor_map.keys["spherical_harmonics_l"] == l)[0]
        ]
        for k, b in zip(l_keys, l_blocks):
            keys.append(tuple(k))
            blocks.append(_block_to_torch(b, structure_i))

    return TensorMap(
        Labels(tensor_map.keys.names, np.asarray(keys, dtype=np.int32)), blocks
    )


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames,
        all_species,
        hypers,
        energies,
        forces=None,
        do_gradients=False,
    ):
        self._all_neighbor_species = Labels(
            names=["species_neighbor"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )
        self._all_center_species = Labels(
            names=["species_center"],
            values=np.array(all_species, dtype=np.int32).reshape(-1, 1),
        )

        self.do_gradients = do_gradients
        self.composition = []

        self.composition_calculator = CompositionFeatures(all_species)
        for frame_i, frame in enumerate(frames):
            self.composition.append(self.compute_composition(frame, frame_i))

        hypers_radial_spectrum = copy.deepcopy(hypers.get("radial_spectrum"))
        self.radial_spectrum = []
        if hypers_radial_spectrum is not None:
            hypers_radial_spectrum["max_angular"] = 0
            self.radial_spectrum_calculator = Calculator(
                SphericalExpansion(**hypers_radial_spectrum)
            )
            self.sum_structures = SumStructures()

            for frame_i, frame in enumerate(frames):
                system = as_torch_system(frame, positions_requires_grad=do_gradients)
                self.radial_spectrum.append(
                    _detach_all_blocks(self.compute_radial_spectrum(system, frame_i))
                )

        else:
            self.radial_spectrum = [None] * len(frames)

        self.spherical_expansions = []
        hypers_spherical_expansion = copy.deepcopy(hypers.get("spherical_expansion"))
        if "radial_per_angular" in hypers_spherical_expansion:
            radial_per_angular = hypers_spherical_expansion.pop("radial_per_angular")

            self.spherical_expansion_calculator = {}
            for l, n in radial_per_angular.items():
                new_hypers = copy.deepcopy(hypers_spherical_expansion)
                new_hypers["max_angular"] = l
                new_hypers["max_radial"] = n
                new_hypers["single_l"] = True
                self.spherical_expansion_calculator[l] = Calculator(
                    SphericalExpansion(**new_hypers)
                )
        else:
            self.spherical_expansion_calculator = Calculator(
                SphericalExpansion(**hypers_spherical_expansion)
            )

        for frame_i, frame in enumerate(frames):
            system = as_torch_system(frame, positions_requires_grad=do_gradients)
            self.spherical_expansions.append(
                _detach_all_blocks(self.compute_spherical_expansion(system, frame_i))
            )

        assert isinstance(energies, torch.Tensor)
        assert energies.shape == (len(frames), 1)
        self.energies = energies

        if forces is not None:
            assert isinstance(forces, list)
            for i, f in enumerate(forces):
                assert isinstance(f, torch.Tensor)
                assert f.shape == (len(frames[i]), 3)

        self.forces = forces
        self._getitemtime = 0
        self._collatetime = 0

    def compute_composition(self, frame, frame_i):
        return self.composition_calculator(
            [frame], np.array([frame_i], dtype=np.int32).reshape(-1, 1)
        )

    def compute_radial_spectrum(self, system, system_i):
        spherical_expansion = self.radial_spectrum_calculator(
            system,
            keep_forward_grad=self.do_gradients,
        )
        
        # TODO: these don't hurt, but are confusing, let's remove them
        spherical_expansion.keys_to_properties(self._all_neighbor_species)
        spherical_expansion.keys_to_properties(self._all_center_species)
        spherical_expansion.components_to_properties("spherical_harmonics_m")
        spherical_expansion.keys_to_properties("spherical_harmonics_l")
        

        return self.sum_structures(_move_to_torch(spherical_expansion, system_i))

    def compute_spherical_expansion(self, system, system_i):
        if isinstance(self.spherical_expansion_calculator, dict):
            spherical_expansion_by_l = {}

            lkeys = []; lblocks = []
            for l, calculator in self.spherical_expansion_calculator.items():

                start = time()                
                spherical_expansion = calculator(
                    system,
                    keep_forward_grad=self.do_gradients,
                )
                """
def _move_to_torch_by_l(tensor_maps, structure_i):
    keys = []
    blocks = []
    for l, tensor_map in tensor_maps.items():
        l_blocks = tensor_map.block(spherical_harmonics_l=l)
        if not hasattr(l_blocks, "__len__"):
            l_blocks = [l_blocks]
        l_keys = tensor_map.keys[
            np.where(tensor_map.keys["spherical_harmonics_l"] == l)[0]
        ]
        for k, b in zip(l_keys, l_blocks):
            keys.append(tuple(k))
            blocks.append(_block_to_torch(b, structure_i))

    return TensorMap(
        Labels(tensor_map.keys.names, np.asarray(keys, dtype=np.int32)), blocks
    )
                """
                lnames = spherical_expansion.keys.names
                #print(spherical_expansion.keys, spherical_expansion.keys.names, l)
                for lk, lb in spherical_expansion:                    
                    if lk["spherical_harmonics_l"]==l:
                        lkeys.append(tuple(lk));
                        lblocks.append(lb.copy())

                print ("calc ", time()-start)
                #start = time()                
                #spherical_expansion.keys_to_samples("species_center")
                #spherical_expansion.keys_to_properties(self._all_neighbor_species)
                #print ("keys_move ", time()-start)
                #spherical_expansion_by_l[l] = spherical_expansion
            start = time()                                            
            spherical_expansion_sel = TensorMap(keys=Labels(names=lnames, values=np.asarray(lkeys, dtype=np.int32)),
                    blocks=lblocks
                    )
            spherical_expansion_sel.keys_to_samples("species_center")
            spherical_expansion_sel.keys_to_properties(self._all_neighbor_species)            
            print("combined move", time()-start)
            return _move_to_torch(spherical_expansion_sel, system_i)
            start = time()
            m2t = _move_to_torch_by_l(spherical_expansion_by_l, system_i)
            print("totorch ", time()-start)
            return m2t
        else:
            spherical_expansion = self.spherical_expansion_calculator(
                system,
                keep_forward_grad=self.do_gradients,
            )
            spherical_expansion.keys_to_samples("species_center")
            spherical_expansion.keys_to_properties(self._all_neighbor_species)

            return _move_to_torch(spherical_expansion, system_i)
#pot, force, stress [[-992.09006198]] (125, 3) (3, 3)

    def __len__(self):
        return len(self.composition)

    def __getitem__(self, idx):
        start = time()
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
        self._getitemtime += time() - start
        return data


def _collate_tensor_map_old(tensors, device):
    keys = tensors[0].keys

    blocks = []
    for key_i, key in enumerate(keys):
        # this assumes that the keys are in the same order in all tensors (which
        # should be fine in this project)
        first_block = tensors[0].block(key_i)
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
            block = tensor.block(key_i)

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


def _collate_tensor_map(tensors, device):
    key_names = tensors[0].keys.names
    sample_names = tensors[0].block(0).samples.names
    if tensors[0].block(0).has_gradient("positions"):
        grad_sample_names = tensors[0].block(0).gradient("positions").samples.names
    unique_keys = set()
    for tensor in tensors:
        unique_keys.update(set(tensor.keys.tolist()))
    unique_keys = [tuple(k) for k in unique_keys]
    unique_keys.sort()
    values_dict = {key: [] for key in unique_keys}
    samples_dict = {key: [] for key in unique_keys}
    properties_dict = {key: None for key in unique_keys}
    components_dict = {key: None for key in unique_keys}
    grad_values_dict = {key: [] for key in unique_keys}
    grad_samples_dict = {key: [] for key in unique_keys}
    grad_components_dict = {key: None for key in unique_keys}
    previous_samples_count = {key: 0 for key in unique_keys}

    for tensor in tensors:
        for key, block in tensor:
            key = tuple(key)
            if components_dict[key] is None:
                # components and properties must be the same for each block of
                # the same key.
                components_dict[key] = block.components
                properties_dict[key] = block.properties
            values_dict[key].append(block.values)

            samples = np.asarray(block.samples.tolist())
            samples_dict[key].append(samples)

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                if grad_components_dict[key] is None:
                    grad_components_dict[key] = gradient.components
                grad_values_dict[key].append(gradient.data)

                grad_samples = np.asarray(gradient.samples.tolist())
                grad_samples[:, 0] += previous_samples_count[key]
                grad_samples_dict[key].append(grad_samples)

            previous_samples_count[key] += samples.shape[0]

    blocks = []
    for key in unique_keys:
        block = TensorBlock(
            values=torch.vstack(values_dict[key]).to(device),
            samples=Labels(
                names=sample_names,
                values=np.asarray(np.vstack(samples_dict[key]), dtype=np.int32),
            ),
            components=components_dict[key],
            properties=properties_dict[key],
        )
        if grad_components_dict[key] is not None:
            block.add_gradient(
                "positions",
                data=torch.vstack(grad_values_dict[key]).to(device),
                components=grad_components_dict[key],
                samples=Labels(
                    names=grad_sample_names,
                    values=np.asarray(
                        np.vstack(grad_samples_dict[key]), dtype=np.int32
                    ),
                ),
            )
        blocks.append(block)

    return TensorMap(Labels(key_names, np.asarray(unique_keys, dtype=np.int32)), blocks)


def _collate_data(device, dataset):
    def do_collate(data):
        start = time()
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
        dataset._collatetime += time() - start
        return composition, radial_spectrum, spherical_expansion, energies, forces

    return do_collate


def create_dataloader(dataset, batch_size, shuffle=True, device="cpu"):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_data(device, dataset),
    )
