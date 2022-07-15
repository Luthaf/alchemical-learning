import copy
from time import time

import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap
from rascaline import SphericalExpansion

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
        values=torch.tensor(block.values).to(dtype=torch.get_default_dtype()),
        samples=samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter in block.gradients_list():
        gradient = block.gradient(parameter)

        assert gradient.samples.names == ("sample", "structure", "atom")

        gradient_samples = (
            gradient.samples.view(dtype=np.int32)
            .reshape(-1, len(gradient.samples.names))
            .copy()
        )
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
        normalization=None,
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
        hypers_radial_spectrum = copy.deepcopy(hypers.get("radial_spectrum", None))
        if hypers_radial_spectrum is not None:
            hypers_radial_spectrum["max_angular"] = 0
            calculator = SphericalExpansion(**hypers_radial_spectrum)
            sum_structures = SumStructures()

            sph_norm = 0.0
            n_env = 0
            for frame_i, frame in enumerate(frames):
                spherical_expansion = calculator.compute(frame)
                spherical_expansion.keys_to_properties(all_center_species)

                # TODO: these don't hurt, but are confusing, let's remove them
                spherical_expansion.components_to_properties("spherical_harmonics_m")
                spherical_expansion.keys_to_properties("spherical_harmonics_l")
                spherical_expansion.keys_to_properties(all_neighbor_species)
                n_env += len(spherical_expansion.block(0).samples)
                for k, b in spherical_expansion:
                    sph_norm += ((b.values**2).sum()).item()
                # sph_structure = summer(spherical_expansion)
                self.radial_spectrum.append(
                    sum_structures(_move_to_torch(spherical_expansion, frame_i))
                )
            if normalization is None:
                self.radial_norm = None
            else:
                if type(normalization) is str and normalization == "automatic":
                    self.radial_norm = 1.0 / np.sqrt(sph_norm / n_env)
                else:
                    self.radial_norm = normalization["radial_spectrum"]
                for r in self.radial_spectrum:
                    for k, b in r:
                        b.values.data *= self.radial_norm
                        if b.has_gradient("positions"):
                            gradient = b.gradient("positions")
                            view = gradient.data.view(gradient.data.dtype)
                            view *= self.radial_norm
        else:
            self.radial_spectrum = [None] * len(frames)

        self.spherical_expansions = []
        hypers_spherical_expansion = copy.deepcopy(
            hypers.get("spherical_expansion", None)
        )
        if "radial_per_angular" in hypers_spherical_expansion:
            radial_per_angular = hypers_spherical_expansion.pop("radial_per_angular")

            calculators = {}
            for l, n in radial_per_angular.items():
                new_hypers = copy.deepcopy(hypers_spherical_expansion)
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
            calculator = SphericalExpansion(**hypers_spherical_expansion)

            for frame_i, frame in enumerate(frames):
                spherical_expansion = calculator.compute(frame)
                spherical_expansion.keys_to_samples("species_center")
                spherical_expansion.keys_to_properties(all_neighbor_species)

                self.spherical_expansions.append(
                    _move_to_torch(spherical_expansion, frame_i)
                )

        if normalization is None:
            self.spherical_expansion_norm = None
        else:
            if type(normalization) is str and normalization == "automatic":
                spex_norm = 0.0
                n_env = 0
                for spex in self.spherical_expansions:
                    n_env += len(spex.block(0).samples)
                    for k, b in spex:
                        spex_norm += ((b.values) ** 2).sum().item()

                self.spherical_expansion_norm = 1.0 / np.sqrt(spex_norm / n_env)
            else:
                self.spherical_expansion_norm = normalization["spherical_expansion"]
            for spex in self.spherical_expansions:
                for k, b in spex:
                    b.values.data *= self.spherical_expansion_norm
                    if b.has_gradient("positions"):
                        gradient = b.gradient("positions")
                        view = gradient.data.view(gradient.data.dtype)
                        view *= self.spherical_expansion_norm

        self.composition = []
        if all_species is not None:
            # composition features are intrinsically normalized
            comp_calc = CompositionFeatures(all_species)
            for frame_i, frame in enumerate(frames):
                comp = comp_calc(
                    [frame], np.array([frame_i], dtype=np.int32).reshape(-1, 1)
                )
                self.composition.append(comp)
        else:
            self.composition = [None] * len(frames)

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
    values_dict = {key: [] for key in unique_keys}
    samples_dict = {key: [] for key in unique_keys}
    properties_dict = {key: None for key in unique_keys}
    components_dict = {key: None for key in unique_keys}
    grad_values_dict = {key: [] for key in unique_keys}
    grad_samples_dict = {key: [] for key in unique_keys}
    grad_components_dict = {key: None for key in unique_keys}
    for tensor in tensors:
        for key, block in tensor:
            key = tuple(key)
            if components_dict[key] is None:
                # components and properties must be the same for each block of
                # the same key.
                components_dict[key] = block.components
                properties_dict[key] = block.properties
            values_dict[key].append(block.values)
            samples_dict[key].append(np.asarray(block.samples.tolist()))
            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                if grad_components_dict[key] is None:
                    grad_components_dict[key] = gradient.components
                grad_values_dict[key].append(gradient.data)
                grad_samples_dict[key].append(np.asarray(gradient.samples.tolist()))
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

    """
    for key_i, key in enumerate(unique_keys):
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
    """
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
