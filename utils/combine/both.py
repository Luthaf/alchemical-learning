import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap


class CombineRadialSpecies(torch.nn.Module):
    def __init__(
        self,
        n_species,
        max_radial,
        n_combined_basis,
        seed=None,
        *,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.n_species = n_species
        self.max_radial = max_radial

        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((max_radial * n_species, n_combined_basis))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineRadialSpecies(
            self.n_species,
            self.max_radial,
            self.combining_matrix.shape[1],
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        n_properties, n_combined_basis = self.combining_matrix.shape

        properties = Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(n_combined_basis)],
                dtype=np.int32,
            ),
        )

        blocks = []
        for _, block in spherical_expansion:
            new_block = TensorBlock(
                values=block.values @ self.combining_matrix,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ self.combining_matrix,
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)

class CombineRadialSpeciesWithAngular(torch.nn.Module):
    def __init__(
        self,
        n_species,
        max_radial,
        max_angular,
        n_combined_basis,
        seed=None,
        *,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.n_species = n_species
        self.max_radial = max_radial
        self.n_angular = max_angular + 1

        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((self.n_angular, max_radial * n_species, n_combined_basis))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineRadialSpeciesWithAngular(
            self.n_species,
            self.max_radial,
            self.n_angular-1,
            self.combining_matrix.shape[2],
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        n_angular, n_properties, n_combined_basis = self.combining_matrix.shape

        properties = Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(n_combined_basis)],
                dtype=np.int32,
            ),
        )

        blocks = []
        for _, block in spherical_expansion:
            l = (block.values.shape[-2] - 1) // 2
            new_block = TensorBlock(
                values=block.values @ self.combining_matrix[l],
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ self.combining_matrix[l],
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)

class CombineRadialSpeciesWithAngularAdaptBasis(torch.nn.Module):
    def __init__(
        self,
        n_species,
        max_radial,
        max_angular,
        combined_basis_per_angular=None,
        n_combined_basis=16,
        seed=None,
        *,
        explicit_combining_matrices=None,
    ):
        super().__init__()
        self.n_species = n_species
        self.max_radial = max_radial
        self.n_angular = max_angular + 1
        self.n_combined_basis = n_combined_basis
        self.combined_basis_per_angular = combined_basis_per_angular
        if self.combined_basis_per_angular is None:
            self.combined_basis_per_angular = {l : n_combined_basis for l in range(self.n_angular)}
        assert len(self.combined_basis_per_angular) == self.n_angular
        # if np.all([max_radial * n_species >= self.combined_basis_per_angular[l] for l in range(self.n_angular)]):
        #     print("Warning: Combiner matrix will do decompression (n_combined_basis > n_radial * n_species)!")
        compress_l = [self.max_radial * self.n_species <= self.combined_basis_per_angular[l] for l in range(self.n_angular)]
        if np.any(compress_l):
            print("Warning: Combiner matrix will do decompression (n_combined_basis > n_radial * n_species)!")
            for l in range(self.n_angular):
                if compress_l[l]:
                    print(f"         l = {l}, n_combined_basis = {self.combined_basis_per_angular[l]}, "\
                        "n_radial  = {self.max_radial}, n_species = {self.n_species})!")

        if explicit_combining_matrices is None:
            if seed is not None:
                torch.manual_seed(seed)
            # self.combining_matrices = [0] * self.n_angular
            # for l, n_basis in combined_basis_per_angular.items():
            #     self.combining_matrices[l] = torch.nn.Parameter(
            #     torch.rand((max_radial * n_species, n_basis))
            # )
            self.combining_matrices = torch.nn.ParameterList([
                torch.nn.Parameter(torch.rand((max_radial * n_species, n_basis)))
                for l, n_basis in combined_basis_per_angular.items()])
        else:
            self.register_buffer("combining_matrices", explicit_combining_matrices)

    def detach(self):
        return CombineRadialSpeciesWithAngularAdaptBasis(
            self.n_species,
            self.max_radial,
            self.n_angular-1,
            self.combined_basis_per_angular,
            self.n_combined_basis,
            # explicit_combining_matrices=[combining_matrix.clone().detach() for combining_matrix in self.combining_matrices],
            explicit_combining_matrices=self.combining_matrices.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        properties = [Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(self.combined_basis_per_angular[l])],
                dtype=np.int32,
            ),
        ) for l in range(self.n_angular)]

        blocks = []
        for _, block in spherical_expansion:
            l = (block.values.shape[-2] - 1) // 2
            new_block = TensorBlock(
                values=block.values @ self.combining_matrices[l],
                samples=block.samples,
                components=block.components,
                properties=properties[l],
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ self.combining_matrices[l],
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)

class CombineRadialSpeciesWithAngularAdaptBasisRadial(torch.nn.Module):
    def __init__(
        self,
        n_species,
        max_radial,
        max_angular,
        radial_per_angular=None,
        combined_basis_per_angular=None,
        n_combined_basis=16,
        seed=None,
        *,
        explicit_combining_matrices=None,
    ):
        super().__init__()
        self.n_species = n_species
        self.max_radial = max_radial
        self.n_angular = max_angular + 1
        self.radial_per_angular = radial_per_angular
        self.n_combined_basis = n_combined_basis
        self.combined_basis_per_angular = combined_basis_per_angular
        if self.combined_basis_per_angular is None:
            self.combined_basis_per_angular = {l : n_combined_basis for l in range(self.n_angular)}
        if self.radial_per_angular is None:
            self.radial_per_angular = {l : max_radial for l in range(self.n_angular)}
        assert len(self.combined_basis_per_angular) == self.n_angular
        assert len(self.radial_per_angular) == self.n_angular
        compress_l = [self.radial_per_angular[l] * self.n_species <= self.combined_basis_per_angular[l] for l in range(self.n_angular)]
        if np.any(compress_l):
            print("Warning: Combiner matrix will do decompression (n_combined_basis > n_radial * n_species)!")
            for l in range(self.n_angular):
                if compress_l[l]:
                    print(f"         l = {l}, n_combined_basis = {self.combined_basis_per_angular[l]}, "\
                        "n_radial  = {self.radial_per_angular[l]}, n_species = {self.n_species})!")

        if explicit_combining_matrices is None:
            if seed is not None:
                torch.manual_seed(seed)
            assert len(self.radial_per_angular) == self.n_angular
            self.combining_matrices = torch.nn.ParameterList([
                torch.nn.Parameter(torch.rand((self.radial_per_angular[l] * n_species, self.combined_basis_per_angular[l])))
                for l in range(self.n_angular)])
        else:
            self.register_buffer("combining_matrices", explicit_combining_matrices)

    def detach(self):
        return CombineRadialSpeciesWithAngularAdaptBasisRadial(
            self.n_species,
            self.max_radial,
            self.n_angular-1,
            self.radial_per_angular,
            self.combined_basis_per_angular,
            self.n_combined_basis,
            # explicit_combining_matrices=[combining_matrix.clone().detach() for combining_matrix in self.combining_matrices],
            explicit_combining_matrices=self.combining_matrices.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l",)
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        properties = [Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(self.combined_basis_per_angular[l])],
                dtype=np.int32,
            ),
        ) for l in range(self.n_angular)]

        blocks = []
        for _, block in spherical_expansion:
            l = (block.values.shape[-2] - 1) // 2
            new_block = TensorBlock(
                values=block.values @ self.combining_matrices[l],
                samples=block.samples,
                components=block.components,
                properties=properties[l],
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ self.combining_matrices[l],
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        return TensorMap(spherical_expansion.keys, blocks)

class CombineRadialSpeciesWithCentralSpecies(torch.nn.Module):
    def __init__(
        self,
        all_species,
        max_radial,
        n_combined_basis,
        seed=None,
        *,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.all_species = all_species
        self.n_species = len(all_species)
        self.max_radial = max_radial
        self.central_species_index = {s: i for i, s in enumerate(all_species)}

        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((self.n_species, max_radial * self.n_species, n_combined_basis))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineRadialSpeciesWithCentralSpecies(
            self.all_species,
            self.max_radial,
            self.combining_matrix.shape[2],
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l","species_center")
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        _, _, n_combined_basis = self.combining_matrix.shape

        properties = Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(n_combined_basis)],
                dtype=np.int32,
            ),
        )

        blocks = []
        for key, block in spherical_expansion:
            cs_i = self.central_species_index[key[1]]
            new_block = TensorBlock(
                values=block.values @ self.combining_matrix[cs_i],
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ self.combining_matrix[cs_i],
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)
        
        new_tensor_map = TensorMap(spherical_expansion.keys, blocks)
        new_tensor_map.keys_to_samples("species_center")
        return new_tensor_map