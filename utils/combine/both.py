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
    #NOTE: here n_pseudo_central_species is gamma_C, not beta_C
    def __init__(
        self,
        all_species,
        max_radial,
        n_combined_basis,
        n_pseudo_central_species=None,
        seed=None,
        *,
        explicit_linear_params=None,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.all_species = all_species
        self.n_species = len(all_species)
        self.max_radial = max_radial
        self.central_species_index = {s: i for i, s in enumerate(all_species)}

        if n_pseudo_central_species is None:
            self.n_pseudo_central_species = self.n_species
        else:
            self.n_pseudo_central_species = min(n_pseudo_central_species, self.n_species)
        
        if self.n_pseudo_central_species == self.n_species:
            self.linear_params = None
        elif explicit_linear_params is None:
            self.linear_params = torch.nn.Parameter(torch.rand((self.n_species, self.n_pseudo_central_species)))
        else:
            self.register_buffer("linear_params", explicit_linear_params)
        
        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((self.n_pseudo_central_species, max_radial * self.n_species, n_combined_basis))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineRadialSpeciesWithCentralSpecies(
            self.all_species,
            self.max_radial,
            self.combining_matrix.shape[2],
            self.n_pseudo_central_species,
            explicit_linear_params=None
                if self.linear_params is None
                else self.linear_params.clone().detach(),
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l", "species_center")
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        _, _, n_combined_basis = self.combining_matrix.shape

        properties = Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(n_combined_basis)],
                dtype=np.int32,
            ),
        )

        if self.linear_params is None:
            cs_combining_matrix = self.combining_matrix
        else:
            cs_combining_matrix = self.combining_matrix.swapaxes(0, 1)
            cs_combining_matrix = (self.linear_params @ cs_combining_matrix).swapaxes(0, 1)

        blocks = []
        for key, block in spherical_expansion:
            cs_i = self.central_species_index[key[1]]
            new_block = TensorBlock(
                values=block.values @ cs_combining_matrix[cs_i],
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")
                new_block.add_gradient(
                    "positions",
                    gradient.data @ cs_combining_matrix[cs_i],
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        new_tensor_map = TensorMap(spherical_expansion.keys, blocks)
        new_tensor_map.keys_to_samples("species_center")
        return new_tensor_map

class CombineSpeciesWithCentralSpecies(torch.nn.Module):
    # This combiner does not support L-dependent combining matrices
    # self.linear_params are common for any L
    def __init__(
        self,
        all_species,
        max_radial,
        n_pseudo_neighbor_species,
        n_pseudo_central_species=None,
        seed=None,
        *,
        explicit_linear_params=None,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.all_species = all_species
        self.n_species = len(all_species)
        self.max_radial = max_radial
        self.n_pseudo_neighbor_species = n_pseudo_neighbor_species
        self.species_remapping = {species: i for i, species in enumerate(all_species)}
        #TODO: extend to L more than 1
        self.l_channels = 1

        if n_pseudo_central_species is None:
            self.n_pseudo_central_species = self.n_species
        else:
            self.n_pseudo_central_species = min(n_pseudo_central_species, self.n_species)
        
        if self.n_pseudo_central_species == self.n_species:
            self.linear_params = None
        elif explicit_linear_params is None:
            self.linear_params = torch.nn.Parameter(torch.rand((self.n_species, self.n_pseudo_central_species)))
        else:
            self.register_buffer("linear_params", explicit_linear_params)

        if explicit_combining_matrix is None:
            if seed is not None:
                torch.manual_seed(seed)
            self.combining_matrix = torch.nn.Parameter(
                torch.rand((self.l_channels, self.n_pseudo_central_species, self.n_species, self.n_pseudo_neighbor_species))
            )
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    def detach(self):
        return CombineSpeciesWithCentralSpecies(
            self.all_species,
            self.max_radial,
            self.n_pseudo_neighbor_species,
            self.n_pseudo_central_species,
            explicit_linear_params=None
                if self.linear_params is None
                else self.linear_params.clone().detach(),
            explicit_combining_matrix=self.combining_matrix.clone().detach(),
        )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l", "species_center")
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        l_channels, n_pseudo_c_species, n_species, n_pseudo_n_species = self.combining_matrix.shape
        n_radial = self.max_radial

        properties = Labels(
            names=["species_neighbor", "n"],
            values=np.array(
                [[-s, n] for s in range(n_pseudo_n_species) for n in range(n_radial)],
                dtype=np.int32,
            ),
        )

        if self.linear_params is None:
            species_combining_matrix = self.combining_matrix
        else:
            species_combining_matrix = self.combining_matrix.swapaxes(1, 2)
            species_combining_matrix = (self.linear_params @ species_combining_matrix).swapaxes(1, 2)

        blocks = []
        for key, block in spherical_expansion:
            cs_i = self.species_remapping[key["species_center"]]
            l = key["spherical_harmonics_l"]
            if l_channels == 1:
                combining_matrix = species_combining_matrix[0]
            else:  # l-dependent combination matrix
                combining_matrix = species_combining_matrix[l]

            # #NOTE: leave making 'properties' here for further extension
            # radial = np.unique(block.properties["n"])
            # n_radial = len(radial)
            # properties = Labels(
            #     names=["species_neighbor", "n"],
            #     values=np.array(
            #         [[-s, n] for s in range(n_pseudo_n_species) for n in radial],
            #         dtype=np.int32,
            #     ),
            # )

            n_samples, n_components, _ = block.values.shape
            data = block.values.reshape(n_samples, n_components, n_species, n_radial)

            data = data.swapaxes(-1, -2)
            data = data @ combining_matrix[cs_i]

            data = data.swapaxes(-1, -2)
            data = data.reshape(n_samples, n_components, n_radial * n_pseudo_n_species)

            new_block = TensorBlock(
                values=data,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")

                n_grad_samples, n_cartesian, n_spherical, _ = gradient.data.shape

                data = gradient.data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_species,
                    n_radial,
                )

                data = data.swapaxes(-1, -2)
                data = data @ combining_matrix[cs_i]

                data = data.swapaxes(-1, -2)
                data = data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_radial * n_pseudo_n_species,
                )

                new_block.add_gradient(
                    "positions",
                    data,
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        new_tensor_map = TensorMap(spherical_expansion.keys, blocks)
        new_tensor_map.keys_to_samples("species_center")
        return new_tensor_map

class CombineRadialNeighSpeciesAndCentralSpecies(torch.nn.Module):
    '''
        M_{alpha_C, alpha_N, n, l, m} => M_{alpha_C, (alpha_N, n), l, m} -> M_{alpha_C, gamma_C, q, l, m} =>

        M_{(alpha_C, gamma_C), q, l, m} -> M_{beta_C, q, l, m}
    '''
    def __init__(
        self,
        all_species,
        max_radial,
        max_angular,
        n_pseudo_central_species,   # beta_C
        n_combining_matrix,         # gamma_C
        radial_per_angular=None,
        combined_basis_per_angular=None,
        n_combined_basis=12,
        linear_params_l_dependent=False,
        combining_matrix_l_dependent=False,
        seed=None,
        *,
        explicit_linear_params=None,
        explicit_combining_matrix=None,
    ):
        super().__init__()
        self.all_species = all_species
        self.n_species = len(all_species)
        self.max_radial = max_radial
        self.n_angular = max_angular + 1
        self.n_combining_matrix = n_combining_matrix
        self.radial_per_angular = radial_per_angular
        self.n_combined_basis = n_combined_basis
        self.linear_params_l_dependent = linear_params_l_dependent,
        self.combining_matrix_l_dependent = combining_matrix_l_dependent,
        self.species_remapping = {species: i for i, species in enumerate(all_species)}
        if seed is not None:
            torch.manual_seed(seed)

        if (combined_basis_per_angular is not None or radial_per_angular is not None) and combining_matrix_l_dependent == False:
            self.combining_matrix_l_dependent = True

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
                    print(f"         l = {l}, n_combined_basis = {self.combined_basis_per_angular[l]}, " \
                        "n_radial  = {self.radial_per_angular[l]}, n_species = {self.n_species})!")
        
        if n_pseudo_central_species is None:
            self.n_pseudo_central_species = self.n_species
        elif n_pseudo_central_species > self.n_species:
            print("Warning: The number of pseudo central species is more than the number of central species (n_pseudo_central_species > n_species)!")
            print("         Set n_pseudo_central_species = n_species = ", self.n_species)
            self.n_pseudo_central_species = self.n_species
        
        if self.n_pseudo_central_species == self.n_species:
            self.linear_params = None
        elif explicit_linear_params is None:
            l_channels = 1
            if self.linear_params_l_dependent:
                l_channels = self.n_angular
            self.linear_params = torch.nn.Parameter(
                torch.rand((l_channels, self.n_species, self.n_combining_matrix, self.n_pseudo_central_species))) # (l+1) x (alpha_C, beta_C, gamma_C)
        else:
            self.register_buffer("linear_params", explicit_linear_params)

        if explicit_combining_matrix is None:
            l_channels = 1
            if self.combining_matrix_l_dependent:
                l_channels = self.n_angular
            self.combining_matrix = torch.nn.ParameterList([
                torch.nn.Parameter(torch.rand((self.n_combining_matrix, self.radial_per_angular[l] * self.n_species, self.combined_basis_per_angular[l])))
                for l in range(l_channels)]) # (l+1) x (gamma_C, (n[l], alpha_N), q[l])
        else:
            self.register_buffer("combining_matrix", explicit_combining_matrix)

    # def detach(self):
    #     return CombineSpeciesWithCentralSpecies(
    #         self.all_species,
    #         self.max_radial,
    #         self.n_pseudo_neighbor_species,
    #         self.n_pseudo_central_species,
    #         explicit_linear_params=None
    #             if self.linear_params is None
    #             else self.linear_params.clone().detach(),
    #         explicit_combining_matrix=self.combining_matrix.clone().detach(),
    #     )

    def forward(self, spherical_expansion: TensorMap):
        assert spherical_expansion.keys.names == ("spherical_harmonics_l", "species_center")
        assert spherical_expansion.property_names == ("species_neighbor", "n")

        # l_channels, n_pseudo_c_species, n_species, n_pseudo_n_species = self.combining_matrix.shape
        # n_radial = self.max_radial

        # properties = Labels(
        #     names=["species_neighbor", "n"],
        #     values=np.array(
        #         [[-s, n] for s in range(n_pseudo_n_species) for n in range(n_radial)],
        #         dtype=np.int32,
        #     ),
        # )
        properties = [Labels(
            names=["combined_basis"],
            values=np.array(
                [[-n] for n in range(self.combined_basis_per_angular[l])],
                dtype=np.int32,
            ),
        ) for l in range(self.n_angular)]

        # ...
        # blocks = []
        # for _, block in spherical_expansion:
        #     l = (block.values.shape[-2] - 1) // 2
        #     new_block = TensorBlock(
        #         values=block.values @ self.combining_matrices[l],
        #         samples=block.samples,
        #         components=block.components,
        #         properties=properties[l],
        #     )

        #     if block.has_gradient("positions"):
        #         gradient = block.gradient("positions")
        #         new_block.add_gradient(
        #             "positions",
        #             gradient.data @ self.combining_matrices[l],
        #             gradient.samples,
        #             gradient.components,
        #         )

        # ...
        # if self.linear_params is None:
        #     cs_combining_matrix = self.combining_matrix
        # else:
        #     cs_combining_matrix = self.combining_matrix.swapaxes(0, 1)
        #     cs_combining_matrix = (self.linear_params @ cs_combining_matrix).swapaxes(0, 1)

        # blocks = []
        # for key, block in spherical_expansion:
        #     cs_i = self.central_species_index[key[1]]
        #     new_block = TensorBlock(
        #         values=block.values @ cs_combining_matrix[cs_i],
        #         samples=block.samples,
        #         components=block.components,
        #         properties=properties,
        #     )

        #     if block.has_gradient("positions"):
        #         gradient = block.gradient("positions")
        #         new_block.add_gradient(
        #             "positions",
        #             gradient.data @ cs_combining_matrix[cs_i],
        #             gradient.samples,
        #             gradient.components,
        #         )

        #     blocks.append(new_block)

        # new_tensor_map = TensorMap(spherical_expansion.keys, blocks)
        # new_tensor_map.keys_to_samples("species_center")
        # return new_tensor_map

        if self.linear_params is None:
            cs2pcs_combining_matrix = self.combining_matrix
        else:
            cs2pcs_combining_matrix = [] # (l+1) x (alpha_C, beta_C, (n[l], alpha_N), q[l])
            for l in range(self.n_angular):
                # (alpha_C, beta_C, (n[l], alpha_N), q[l])
                # lp (alpha_C, beta_C, gamma_C)
                # cm (gamma_C, (n[l], alpha_N), q[l])

                matrix = torch.zeros(size=(self.n_species, self.n_pseudo_central_species, self.radial_per_angular[l] * self.n_species, self.combined_basis_per_angular[l]))
                # matrix = 
            species_combining_matrix = self.combining_matrix.swapaxes(1, 2)
            species_combining_matrix = (self.linear_params @ species_combining_matrix).swapaxes(1, 2)

        blocks = []
        for key, block in spherical_expansion:
            cs_i = self.species_remapping[key["species_center"]]
            l = key["spherical_harmonics_l"]
            if l_channels == 1:
                combining_matrix = species_combining_matrix[0]
            else:  # l-dependent combination matrix
                combining_matrix = species_combining_matrix[l]

            # #NOTE: leave making 'properties' here for further extension
            # radial = np.unique(block.properties["n"])
            # n_radial = len(radial)
            # properties = Labels(
            #     names=["species_neighbor", "n"],
            #     values=np.array(
            #         [[-s, n] for s in range(n_pseudo_n_species) for n in radial],
            #         dtype=np.int32,
            #     ),
            # )

            n_samples, n_components, _ = block.values.shape
            data = block.values.reshape(n_samples, n_components, n_species, n_radial)

            data = data.swapaxes(-1, -2)
            data = data @ combining_matrix[cs_i]

            data = data.swapaxes(-1, -2)
            data = data.reshape(n_samples, n_components, n_radial * n_pseudo_n_species)

            new_block = TensorBlock(
                values=data,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )

            if block.has_gradient("positions"):
                gradient = block.gradient("positions")

                n_grad_samples, n_cartesian, n_spherical, _ = gradient.data.shape

                data = gradient.data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_species,
                    n_radial,
                )

                data = data.swapaxes(-1, -2)
                data = data @ combining_matrix[cs_i]

                data = data.swapaxes(-1, -2)
                data = data.reshape(
                    n_grad_samples,
                    n_cartesian,
                    n_spherical,
                    n_radial * n_pseudo_n_species,
                )

                new_block.add_gradient(
                    "positions",
                    data,
                    gradient.samples,
                    gradient.components,
                )

            blocks.append(new_block)

        new_tensor_map = TensorMap(spherical_expansion.keys, blocks)
        new_tensor_map.keys_to_samples("species_center")
        return new_tensor_map