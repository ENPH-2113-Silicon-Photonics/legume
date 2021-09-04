import legume
from typing import Sequence, Literal, Tuple, List
import autograd.numpy as np
import matplotlib.pyplot as plt
import itertools
from chickpea.sbl import ShapeBuilder
from legume.backend import backend as bd


class CrystalTopology:
    """
    Topology Constructor for optimization
    """

    def __init__(self):
        pass

    def crystal(self, params: Sequence[float]) -> legume.phc.PhotCryst:
        """
        Return a photonic crystal topology given input parameters
        @param params:
        @return:
        """
        raise (NotImplementedError("Method must be implemented by subclass."))

    def get_base_crystal(self) -> legume.phc.PhotCryst:
        """
        Return basic primitive cell defining bulk of crystal.
        @return: legume.phc.PhotCryst object defining primitive crystal.
        """
        raise (NotImplementedError("Method must be implemented by subclass."))

    def get_param_vector(self):
        """
        @return: Default parameter vector that can be passed to self.crystal method.
        """
        raise (NotImplementedError("Method must be implemented by subclass."))

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        @return: Bounds of parameters
        """
        raise (NotImplementedError("Method must be implemented by subclass."))


class PhotonicCrystalTopologyBuilder(CrystalTopology):

    def __init__(self, type: Literal['hexagonal', 'square'], supercell_size: Tuple[int, int], thickness: float,
                 radius: float, eps_b: float, eps_circ, eps_l: float = 1, eps_u: float = 1, h_ratio: float=1, v_ratio: float=1):
        super().__init__()
        self.eps_b = eps_b
        self.eps_circ = eps_circ
        self.thickness = thickness
        self.radius = radius
        self.type = type.lower()
        self._supercell_size = supercell_size

        self.dist_bound = [-(1 / 2 - self.radius), 1 / 2 - self.radius]
        self.rad_bound = [0.85, 1]

        eps_ratio = self.eps_circ / self.eps_b

        self.eps_bound = [min(eps_ratio, 1 / eps_ratio), 1.5]

        self.eps_l = eps_l
        self.eps_u = eps_u

        Nx, Ny = self._supercell_size

        ix, iy = bd.meshgrid(bd.array(range(Nx), dtype=bd.int_), bd.array(range(Ny), dtype=bd.int_))
        self.init_grid = (ix.T, iy.T)

        if self.type == 'hexagonal':
            if Ny % 2 == 1:
                raise ValueError("For periodicity Y periods of hex lattice should always be even.")

            self.pos_grid = bd.array([(self.init_grid[0] + 0.5 * self.init_grid[1]) % Nx,
                                      bd.sqrt(3) / 2 * (self.init_grid[1])])

            lattice = legume.Lattice([supercell_size[0], 0], [0, (supercell_size[1]) * np.sqrt(3) / 2])
            self._lattice = lattice

        elif self.type == 'square':
            self.xgrid, self.ygrid = bd.meshgrid(bd.array(range(Nx), dtype=bd.float64),
                                                 bd.array(range(Ny), dtype=bd.float64))
            self.pos_grid = (self.xgrid.T, self.ygrid.T)

            lattice = legume.Lattice([supercell_size[0], 0], [0, supercell_size[1]])
            self._lattice = lattice
        elif self.type == 'rectangular':
            self.xgrid, self.ygrid = np.meshgrid(np.array(range(Nx)*self.h_ratio, dtype=np.float64),
                                                 np.array(range(Ny)*self.v_ratio, dtype=np.float64))
            self.pos_grid = (self.xgrid.T, self.ygrid.T)

            lattice = legume.Lattice([supercell_size[0]*self.h_ratio, 0], [0, supercell_size[1]*self.v_ratio])
            self._lattice = lattice
        else:
            raise ValueError("Type must be hexagonal or square")

        self.hole_grid = bd.tile(bd.expand_dims(bd.array([eps_circ, radius]), axis=0), (Nx, Ny, 1))

        self.symmetry = 'None'
        self.sym_mask = bd.ones((Nx, Ny))
        self.sym_cell_shape = supercell_size

        self.x_bounds = [self.dist_bound] * Nx * Ny
        self.y_bounds = [self.dist_bound] * Nx * Ny
        self.rad_bounds = [self.rad_bound] * Nx * Ny
        self.eps_bounds = [self.eps_bound] * Nx * Ny

    def replace_hole(self, coord, eps=None, radius=None):
        hole_list = self.hole_grid.tolist()

        if self.type == 'hexagonal' and coord[1] < 0:
            coord = self.hex_coord_convert(coord)
        if eps is None:
            eps = hole_list[coord[0]][coord[1]][0]

        if radius is None:
            radius = hole_list[coord[0]][coord[1]][1]

        hole_list[coord[0]][coord[1]] = [eps, radius]

        self.hole_grid = bd.array(hole_list)

    def cut_waveguides(self, rows=None, cols=None, eps=None, radius=0):
        if cols is not None:
            if self.type == 'hexagonal':
                raise ValueError("Cutting waveguides along columns of hexagonal crystal is not supported.")

            if type(cols) is int:
                col = bd.array(self.init_grid)[:, cols, :]

                for coord in col.T:
                    self.replace_hole(coord, eps=eps, radius=radius)
            else:
                for col in cols:
                    col = bd.array(self.init_grid)[:, col, :]

                    for coord in col.T:
                        self.replace_hole(coord, eps=eps, radius=radius)

        if rows is not None:
            if type(rows) is int:
                row = bd.array(self.init_grid)[:, :, rows]

                for coord in row.T:
                    self.replace_hole(coord, eps=eps, radius=radius)
            else:
                for row in rows:
                    row = bd.array(self.init_grid)[:, :, row]

                    for coord in row.T:
                        self.replace_hole(coord, eps=eps, radius=radius)

    def introduce_point_defect(self):
        raise NotImplementedError("This feature is not implemented")

    def get_base_crystal(self) -> legume.phc.phc:

        lattice = legume.Lattice(self.type)
        cryst = legume.PhotCryst(lattice, eps_l=self.eps_l, eps_u=self.eps_u)
        cryst.add_layer(d=self.thickness, eps_b=self.eps_b)
        cryst.add_shape(legume.Circle(x_cent=0, y_cent=0, r=self.radius, eps=self.eps_circ))
        return cryst

    def hex_coord_convert(self, coord):
        if self.type == 'hexagonal' and coord[1] < 0:
            return (
                int(coord[0] - self._supercell_size[1] / 2) % self._supercell_size[0],
                coord[1] % self._supercell_size[1])
        else:
            return coord

    def set_symmetry_and_param_mask(self, symmetry=None, x_mask=None, y_mask=None, rad_mask=None, eps_mask=None):
        Nx, Ny = self._supercell_size

        symmetry = symmetry.lower()
        if symmetry == 'none' or symmetry is None:
            self.sym_mask = bd.ones((Nx, Ny))

        elif self.type == 'square':
            if symmetry == 'x_mirror':
                shape = (Nx // 2, Ny)
                if Nx % 2 == 0:
                    self.sym_mask = bd.vstack((bd.ones(shape), bd.zeros(shape)))
                else:
                    shape = ((Nx - 1) // 2, Ny)
                    self.sym_mask = bd.vstack((bd.ones(shape), 1 / 2 * bd.ones((1, shape[1])), bd.zeros(shape)))

                self.sym_cell_shape = ((Nx + 1) // 2, Ny)

            elif symmetry == 'y_mirror':
                shape = (Nx, Ny // 2)
                if Ny % 2 == 0:
                    self.sym_mask = bd.hstack((bd.zeros(shape), bd.ones(shape)))
                else:
                    self.sym_mask = bd.hstack((bd.zeros(shape), 1 / 2 * bd.ones((shape[0], 1)), bd.ones(shape)))
                self.sym_cell_shape = (Nx, (Ny + 1) // 2)

            elif symmetry == 'dihedral':
                shape = (Nx // 2, Ny // 2)
                if Nx % 2 == 0 and Ny % 2 == 0:
                    col1 = bd.hstack((bd.zeros(shape), bd.ones(shape)))
                    col2 = bd.hstack((bd.zeros(shape), bd.zeros(shape)))

                    self.sym_mask = bd.vstack((col1, col2))
                elif Nx % 2 == 0 and Ny % 2 == 1:
                    col1 = bd.hstack((bd.zeros(shape), 1 / 2 * bd.ones((shape[0], 1)), bd.ones(shape)))
                    col2 = bd.hstack((bd.zeros(shape), bd.zeros((shape[0], 1)), bd.zeros(shape)))
                    self.sym_mask = bd.vstack((col1, col2))

                elif Nx % 2 == 1 and Ny % 2 == 0:
                    col1 = bd.hstack((bd.zeros(shape), bd.ones(shape)))
                    col2 = bd.hstack((bd.zeros((1, shape[1])), 1 / 2 * bd.ones((1, shape[1]))))
                    col3 = bd.hstack((bd.zeros(shape), bd.zeros(shape)))

                    self.sym_mask = bd.vstack((col1, col2, col3))

                elif Nx % 2 == 1 and Ny % 2 == 1:
                    col1 = bd.hstack((bd.zeros(shape), 1 / 2 * bd.ones((shape[0], 1)), bd.ones(shape)))
                    col2 = bd.hstack((bd.zeros((1, shape[1])), 1 / 4 * bd.ones((1, 1)), 1 / 2 * bd.ones((1, shape[1]))))
                    col3 = bd.hstack((bd.zeros(shape), bd.zeros((shape[0], 1)), bd.zeros(shape)))

                    self.sym_mask = bd.vstack((col1, col2, col3))

                self.sym_cell_shape = ((Nx + 1) // 2, (Ny + 1) // 2)

        if x_mask is None:
            x_mask = bd.ones(self.sym_cell_shape)
        if y_mask is None:
            y_mask = bd.ones(self.sym_cell_shape)
        if rad_mask is None:
            rad_mask = bd.ones(self.sym_cell_shape)
        if eps_mask is None:
            eps_mask = bd.ones(self.sym_cell_shape)

        param_mask = bd.array([x_mask, y_mask, rad_mask, eps_mask])

        if param_mask.shape[1:] != self.sym_cell_shape:
            raise ValueError("Parameter mask shape not equal to symmetry cell")

        self.x_bounds = [[[0, 0], self.dist_bound][val] for val in bd.ravel(param_mask[0])]
        self.y_bounds = [[[0, 0], self.dist_bound][val] for val in bd.ravel(param_mask[1])]
        self.rad_bounds = [[[1, 1], self.rad_bound][val] for val in bd.ravel(param_mask[2])]
        self.eps_bounds = [[[1, 1], self.eps_bound][val] for val in bd.ravel(param_mask[3])]

        self.symmetry = symmetry

    def set_base_bounds(self, dist, rad, eps):

        self.dist_bound = dist
        self.rad_bound = rad
        self.eps_bound = eps

    def _center(self, arrays):
        Nx, Ny = self._supercell_size

        return [bd.roll(array, (bd.int_(bd.ceil(Nx / 2)), bd.int_(bd.ceil(Ny / 2))), axis=(0, 1)) for array in arrays]

    def _uncenter(self, arrays):
        Nx, Ny = self._supercell_size
        return [bd.roll(array, (-bd.int_(bd.ceil(Nx / 2)), -bd.int_(bd.ceil(Ny / 2))), axis=(0, 1)) for array in arrays]

    def _apply_symmetry(self, dx, dy, eps, rad):

        dx_, dy_, eps_, rad_ = self._center([dx, dy, eps, rad])

        dx_masked = dx_ * self.sym_mask
        dy_masked = dy_ * self.sym_mask
        eps_masked = eps_ * self.sym_mask
        rad_masked = rad_ * self.sym_mask
        if self.symmetry == 'None' or self.symmetry is None:
            return dx, dy, eps, rad

        elif self.type == 'square':
            if self.symmetry == 'x_mirror':
                dx_ = dx_masked - bd.flip(dx_masked, 0)
                dy_ = dy_masked + bd.flip(dy_masked, 0)
                eps_ = eps_masked + bd.flip(eps_masked, 0)
                rad_ = rad_masked + bd.flip(rad_masked, 0)
            elif self.symmetry == 'y_mirror':
                dx_ = dx_masked + bd.flip(dx_masked, 1)
                dy_ = dy_masked - bd.flip(dy_masked, 1)
                eps_ = eps_masked + bd.flip(eps_masked, 1)
                rad_ = rad_masked + bd.flip(rad_masked, 1)
            elif self.symmetry == 'Dihedral':
                dx_ = dx_masked + bd.flip(dx_masked, 1) - bd.flip(dx_masked, 0) - bd.flip(dy_masked, (1, 0))
                dy_ = dy_masked - bd.flip(dy_masked, 1) + bd.flip(dy_masked, 0) - bd.flip(dy_masked, (1, 0))
                eps_ = eps_masked + bd.flip(eps_masked, 1) + bd.flip(dy_masked, 0) + bd.flip(dy_masked, (1, 0))
                rad_ = rad_masked + bd.flip(rad_masked, 1) + bd.flip(dy_masked, 0) + bd.flip(dy_masked, (1, 0))
            else:
                raise ValueError("This symmetry not available for squares.")

        dx, dy, eps, rad = self._uncenter([dx_, dy_, eps_, rad_])

        return dx, dy, eps, rad

    def get_bounds(self):
        return self.x_bounds + self.y_bounds + self.rad_bounds + self.eps_bounds

    def get_param_shape(self, ):
        raise NotImplementedError

    def crystal_structured(self, dx: Sequence[float] = None,
                dy: Sequence[float] = None,
                frads: Sequence[float] = None,
                feps: Sequence[float] = None) -> legume.phc.phc:
        """
        Construct a photonic crystal crystal object of topology specified in constructor

        @param dx: array of displacement of holes in the x direction. Length of array must be self.num_holes
        @param dy: array of displacement of holes in the y direction. Length of array must be self.num_holes
        @param rads: array of ratios to radius, hole size will be scaled by rads. Length of array must be self.num_holes

        @return: photonic crystal crystal object
        """
        Nx, Ny = self._supercell_size

        if dx is None:
            dx = bd.zeros(self.sym_cell_shape)
        if dy is None:
            dy = bd.zeros(self.sym_cell_shape)
        if frads is None:
            frads = bd.ones(self.sym_cell_shape)
        if feps is None:
            feps = bd.ones(self.sym_cell_shape)

        dx, dy, frads, feps = self._apply_symmetry(dx, dy, frads, feps)

        cryst = legume.PhotCryst(self._lattice, eps_l=self.eps_l, eps_u=self.eps_u)
        cryst.add_layer(d=self.thickness, eps_b=self.eps_b)

        # Add holes with double mirror symmetry.
        for i in range(Nx):
            for j in range(Ny):
                eps = self.hole_grid[i, j, 0] * feps[i, j]
                rad = self.hole_grid[i, j, 1] * frads[i, j]
                x_cent = self.pos_grid[0][i, j] + dx[i, j]
                y_cent = self.pos_grid[1][i, j] + dy[i, j]

                if rad == 0.0:
                    continue
                else:
                    cryst.add_shape(legume.Circle(x_cent=x_cent, y_cent=y_cent, r=rad, eps=eps))
        return cryst

    def crystal(self, params: Sequence[float]) -> legume.phc.phc:
        """
        Construct a photonic crystal crystal object of topology specified in constructor.
        Parameter list compatible with Autograd and

        @param params: flat bd.array of parameters in the block form [dx, dy, frads, feps].

        Each array flattened in row major order. (bd.ravel)

        @return: photonic crystal crystal object
        """
        Nx, Ny = self._supercell_size

        param_length = self.sym_cell_shape[0] * self.sym_cell_shape[1]
        dx = params[0:param_length].reshape(self.sym_cell_shape)
        dy = params[param_length:2 * param_length].reshape(self.sym_cell_shape)
        frads = params[param_length:3 * param_length].reshape(self.sym_cell_shape)
        feps = params[3 * param_length:].reshape(self.sym_cell_shape)

        dx, dy, frads, feps = self._apply_symmetry(dx, dy, frads, feps)

        cryst = legume.PhotCryst(self._lattice, eps_l=self.eps_l, eps_u=self.eps_u)
        cryst.add_layer(d=self.thickness, eps_b=self.eps_b)

        for i in range(Nx):
            for j in range(Ny):
                eps = self.hole_grid[i, j, 0] * feps[i, j]
                rad = self.hole_grid[i, j, 1] * frads[i, j]
                x_cent = self.pos_grid[0][i, j] + dx[i, j]
                y_cent = self.pos_grid[1][i, j] + dy[i, j]

                if rad == 0.0:
                    continue
                else:
                    cryst.add_shape(legume.Circle(x_cent=x_cent, y_cent=y_cent, r=rad, eps=eps))
        return cryst

    def get_start_parameters(self):
        """
        @return: Returns a list of appopriate shape for starting parameters
        """
        param_length = self.sym_cell_shape[0] * self.sym_cell_shape[1]

        return bd.array([0] * param_length + [0] * param_length + [1] * param_length + [1] * param_length)

    def display_coordinate_grid(self, use_neg=True, fig=None, axis=None):
        """
        Displays the locations of shapes in the lattice with index coordinates overlaid.

        :param use_neg: If true displays negative coordinates close to 0.
                            Note that modular relation is broken for a hexagonal lattice.
                            That is (0,-1) == (-Ny/2 % Nx,Nx-1) != (0, Nx-1).

                            This accounts for the fact that we have a hex lattice embedded in a
                            rectangular superlattice.

                        If false displays true coordinates.

                        Both coordinate systems can be used to reference holes.
        :return:
        """
        spatial_period = self._lattice.a1 + self._lattice.a2
        ax, ay = spatial_period

        if fig is None and axis is None:
            fig, axis = plt.subplots(figsize=self._supercell_size)
        elif fig is None or axis is None:
            raise ValueError("Must have both or neither of fig and axis be None.")

        axis.scatter(bd.mod(self.pos_grid[0] + ax / 2, ax) - ax / 2,
                     bd.mod(self.pos_grid[1] + ay / 2, ay) - ay / 2,
                     s=500,
                     cmap="Greys")

        axis.arrow(0, 0, self.pos_grid[0][1, 0], self.pos_grid[1][1, 0], width=0.1, alpha=1, color='red',
                   length_includes_head=True)
        axis.arrow(0, 0, self.pos_grid[0][0, 1], self.pos_grid[1][0, 1], width=0.1, alpha=1, color='red',
                   length_includes_head=True)
        axis.scatter(0, 0, s=100, c='red')

        if use_neg:
            iterable = itertools.product(range(-self._supercell_size[0] // 2, self._supercell_size[0] // 2),
                                         range(-self._supercell_size[1] // 2, self._supercell_size[1] // 2))
        else:
            iterable = itertools.product(range(self._supercell_size[0]), range(self._supercell_size[1]))

        for coord in iterable:
            if self.type == 'hexagonal':
                i_, j_ = self.hex_coord_convert(coord)
            else:
                i_, j_ = coord
            axis.text(bd.mod(self.pos_grid[0][i_, j_] + ax / 2, ax) - ax / 2 - 0.5,
                      bd.mod(self.pos_grid[1][i_, j_] + ay / 2 + 0.2, ay) - ay / 2, "(%d,%d)" % coord)

        return fig, axis


class GeneralizedPHCTopologyBuilder(CrystalTopology):

    def __init__(self, lattice_type: Literal['hexagonal', 'square', 'custom'], shape: ShapeBuilder, supercell_size: Tuple[int, int], thickness: float, eps_shape: float,
                 eps_b: float, a: float =1, eps_l: float = 1, eps_u: float = 1, custom_lattice_vectors = None):


        self.type = lattice_type.lower()
        self._supercell_size = supercell_size


        self.eps_b = eps_b
        self.eps_shape = eps_shape

        self.thickness = thickness
        self.shape = shape
        self.shape_parameters = shape.parameters

        eps_ratio = self.eps_shape / self.eps_b

        self.eps_bound = [min(eps_ratio, 1 / eps_ratio), max(eps_ratio, 1 / eps_ratio)]

        self.eps_l = eps_l
        self.eps_u = eps_u

        self.Nx, self.Ny = self._supercell_size


        if self.type == 'hexagonal':
            self.lattice_vectors = bd.array([[1, 0], [1 / 2, bd.sqrt(3) / 2]]) * a

        elif self.type == 'square':
            self.lattice_vectors = bd.array([[1, 0], [0, 1]]) * a
        elif self.type == 'custom':
            if custom_lattice_vectors is not None:
                self.lattice_vectors = custom_lattice_vectors
            else:
                raise ValueError("For custom type custom_lattice_vectors must be defined.")
        else:
            raise ValueError("Type must be custom, square or hexagonal")
        lattice = legume.Lattice(self.lattice_vectors[0] * supercell_size[0],
                                 self.lattice_vectors[1] * supercell_size[1], reduce_lattice=False)

        ix, iy = bd.meshgrid(bd.array(range(self.Nx), dtype=bd.int_), bd.array(range(self.Ny), dtype=bd.int_))

        self.init_grid = (ix.T, iy.T)

        ix = ix.T.reshape(self.Nx, self.Ny, 1)
        iy = iy.T.reshape(self.Nx, self.Ny, 1)

        self.pos_grid = bd.array(ix * self.lattice_vectors[0] + iy * self.lattice_vectors[1])

        self._lattice = lattice

        shape_grid = dict()
        for param in self.shape_parameters:
            shape_grid[param] = [[self.shape.defaults[param]] * self.Ny for i in range(self.Nx)]

        self.shape_grid = shape_grid
        self.sym_cell_shape = supercell_size

    def update_shape(self, coord, **shape_parameters):
        for param in shape_parameters:
            self.shape_grid[param][coord[0]][coord[1]] = shape_parameters[param]

    def update_row_col(self, rows=None, cols=None, **shape_parameters):
        if cols is not None:
            if type(cols) is int:
                col = bd.array(self.init_grid)[:, cols, :]

                for coord in col.T:
                    self.update_shape(coord, **shape_parameters)
            else:
                for col in cols:
                    col = bd.array(self.init_grid)[:, col, :]

                    for coord in col.T:
                        self.update_shape(coord, **shape_parameters)

        if rows is not None:
            if type(rows) is int:
                row = bd.array(self.init_grid)[:, :, rows]

                for coord in row.T:
                    self.update_shape(coord, **shape_parameters)
            else:
                for row in rows:
                    row = bd.array(self.init_grid)[:, :, row]

                    for coord in row.T:
                        self.update_shape(coord, **shape_parameters)


    def get_base_crystal(self) -> legume.phc.phc:

        lattice = legume.Lattice(self.lattice_vectors[0], self.lattice_vectors[1])
        cryst = legume.PhotCryst(lattice, eps_l=self.eps_l, eps_u=self.eps_u)
        cryst.add_layer(d=self.thickness, eps_b=self.eps_b)
        self.shape.place_shape(cryst, eps=self.eps_shape, x=0, y=0)
        return cryst

    def crystal(self, params: bd.ndarray) -> legume.phc.phc:
        """
        Construct a photonic crystal crystal object of topology specified in constructor.
        Parameter list compatible with Autograd and

        @param params: flat bd.array of parameters in the block form [dx, dy, feps, ...<shape_parameters>... ].

        Each array flattened in row major order. (bd.ravel)

        @return: photonic crystal crystal object
        """
        Nx, Ny = self._supercell_size

        param_length = self.sym_cell_shape[0] * self.sym_cell_shape[1]

        dx = params[0:param_length].reshape(self.sym_cell_shape)
        dy = params[param_length:2 * param_length].reshape(self.sym_cell_shape)
        feps = params[2 * param_length:3 * param_length].reshape(self.sym_cell_shape)

        parameter_dict_ = dict()
        i = 3
        for key in self.shape_parameters:
            param_dims = self.shape.parameter_dims[key]
            parameter_dict_.update({key: params[i * param_length:(i + param_dims) * param_length].reshape(
                (self.sym_cell_shape[0], self.sym_cell_shape[1], param_dims))})
            i += self.shape.parameter_dims[key]

        parameter_dict = parameter_dict_

        cryst = legume.PhotCryst(self._lattice, eps_l=self.eps_l, eps_u=self.eps_u)
        cryst.add_layer(d=self.thickness, eps_b=self.eps_b)

        for i in range(Nx):
            for j in range(Ny):
                x, y = self.pos_grid[i, j]
                passed_parameters_ = dict()
                for key in self.shape_parameters:
                    passed_parameters_.update({key: parameter_dict[key][i, j]})
                passed_parameters = passed_parameters_

                self.shape.place_shape(cryst, eps=feps[i, j] * self.eps_shape, x=x + dx[i, j], y=y + dy[i, j],
                                       **passed_parameters)

        return cryst

    def get_param_vector(self):

        param_length = self.sym_cell_shape[0] * self.sym_cell_shape[1]
        basic_params = bd.hstack((bd.zeros(2 * param_length), bd.ones(param_length)))
        shape_params = bd.hstack([bd.ravel(self.shape_grid[param], order='C') for param in self.shape_parameters])

        return bd.hstack((basic_params, shape_params))

    def get_bounds(self, x_bounds=(0, 0), y_bounds=(0, 0), eps_bounds=(1, 1)):

        param_length = self.sym_cell_shape[0] * self.sym_cell_shape[1]

        basic_bounds = [x_bounds] * param_length + [y_bounds] * param_length + [eps_bounds] * param_length

        bounds_dict = self.shape.bounds

        shape_bounds = []
        for param in self.shape_parameters:
            shape_bounds = shape_bounds + bounds_dict[param] * param_length
        return basic_bounds + shape_bounds
