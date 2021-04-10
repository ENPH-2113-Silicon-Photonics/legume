# %%

import numpy as np
import legume
from typing import Sequence, Literal, Tuple, Union


# %%

# The idea is to create a class that automatically defines cavity geometries for you.
# Starting with the only changes to the cavity being shifts.
# All of our crystals, including the shifts, have mirror symmetry about the x and y axes.

class PhotonicCrystalCavity:

    # dx and dy should be optional arguments
    def __init__(self, crystal: Literal['H', 'L'], supercell_size: Tuple[int, int],
                 thickness: float, radius: float,
                 eps: float, n: int, m: int = None):
        """
        Sets the topology of a photonic crystal cavity as a Lm-n crystal or a Hn crystal.

        Lm-n crystals are crystals with n holes removed in a line and m holes added.
        Hn crystals are holes with n hexagonal rings of holes removed around center.

        Both crystals are based on hexagonal lattice of holes with unit lattice constant.

        Provides constructor for these topologies allowing for perturbations of radius and placement of
        individual holes in the lattice. Perturbations maintain double mirror symmetry along the X and Y axis.

        @param crystal:  A string that's either 'H' or 'L', depending on what the desired cavity type is


        @param supercell_size: The number of periods of the unperturbed lattice in the X and Y directions.

        @param thickness: The thickness of the slab.

        @param radius: Base radius of holes in units of the lattice constant

        @param eps: The material permittivity of the material (holes are eps=1)

        @param n: For 'L' type crystals represents the number of holes removed along the X axis.
                  For 'H' type crystals represents the number of hexagonal rings of holes removed around origin.

        @param m: For 'L' for L type crystal represents the number of holes to replace the 'n' removed holes.
                  m should not be too much greater then n or holes will overlap.

                  For 'H' type crystal represents nothing. Defaults to None
        """
        if crystal == 'L' and m is None:
            raise ValueError("L crystal requires m parameter defined")
        if crystal == 'H':
            m = None

        self.crystal = [crystal, n, m]
        self._supercell_size = supercell_size

        lattice = legume.Lattice([supercell_size[0], 0], [0, supercell_size[1] * np.sqrt(3) / 2])
        self._lattice = lattice

        self.thickness = thickness
        self.eps = eps
        self.radius = radius

        n, m = self.crystal[1:]
        ctype = self.crystal[0]

        Nx, Ny = self._supercell_size
        xp, yp = [], []
        nx, ny = Nx // 2 + 1, Ny // 2 + 1

        if n == 0 or (ctype == 'L' and n % 2 == 0):
            for iy in range(ny):
                for ix in range(nx):
                    xp.append(ix + ((iy + 1) % 2) * 0.5)
                    yp.append(iy * np.sqrt(3) / 2)
        else:
            for iy in range(ny):
                for ix in range(nx):
                    xp.append(ix + (iy % 2) * 0.5)
                    yp.append(iy * np.sqrt(3) / 2)

        def remove_holes(xp, yp, Nx, Ny, ctype, m, n):
            # check if crystal valid
            if (n + 1) // 2 >= min((Nx + 1) // 2, (Ny + 1) // 2):
                print("cavity invalid - use a bigger _supercell_size")
                return xp, yp

            # not sure if necessary
            if n == 0:
                return xp, yp

            elif ctype == 'L':
                # remove n holes:
                xremoved = xp.copy()[(n + 1) // 2:]
                yremoved = yp.copy()[(n + 1) // 2:]

                # fill with m holes:
                xfill = list(np.linspace(-xremoved[0], xremoved[0], num=m + 2, endpoint=True)[(m + 2) // 2: -1])
                yfill = list(np.zeros((m + 1) // 2))

                xnew = xfill + xremoved
                ynew = yfill + yremoved
                return xnew, ynew

            elif ctype == 'H':
                xnew = xp.copy()
                ynew = yp.copy()

                for i in range(n):
                    ind = (n - 1 - i) * (Nx // 2) + n - i - 1
                    num = (n + 1 + i) // 2
                    del xnew[ind:ind + num]
                    del ynew[ind:ind + num]
                return xnew, ynew

            else:
                print("invalid crystal type")
                return xp, yp

        self.xp, self.yp = remove_holes(xp, yp, Nx, Ny, ctype, m, n)
        self._num_holes = len(self.xp)
        # FIGURE OUT HOW TO THROW EXCEPTIONS / DEAL WITH BAD INPUTS

    def __repr__(self):
        if self.crystal[2] is None:
            return "PhotonicCrystalCavity(%s, %s, %f, %f, %f, %d)" % (self.crystal[0], str(self._supercell_size),
                                                                      self.thickness, self.radius, self.eps,
                                                                      self.crystal[1])

        else:
            return "PhotonicCrystalCavity(%s, %s, %f, %f, %f, %d, %d)" % (self.crystal[0], str(self._supercell_size),
                                                                      self.thickness, self.radius, self.eps,
                                                                      self.crystal[1], self.crystal[2])

    def cavity(self, dx: Sequence[int] = None, dy: Sequence[int] = None, rads: Sequence[int] = None) -> legume.phc.phc:
        """
        Construct a photonic crystal cavity object of topology specified in constructor

        @param dx: array of displacement of holes in the x direction. Length of array must be self.num_holes
        @param dy: array of displacement of holes in the y direction. Length of array must be self.num_holes
        @param rads: array of ratios to radius, hole size will be scaled by rads. Length of array must be self.num_holes

        @return: photonic crystal cavity object
        """
        Nx, Ny = self._supercell_size
        nx, ny = Nx // 2 + 1, Ny // 2 + 1

        if dx is None:
            dx = np.zeros((self._num_holes,))

        if dy is None:
            dy = np.zeros((self._num_holes,))

        if rads is None:
            rads = np.ones((self._num_holes,))

        cryst = legume.PhotCryst(self._lattice)

        cryst.add_layer(d=self.thickness, eps_b=self.eps)

        # Add holes with double mirror symmetry.
        for ic, x in enumerate(self.xp):
            yc = self.yp[ic] if self.yp[ic] == 0 else self.yp[ic] + dy[ic]
            xc = x if x == 0 else self.xp[ic] + dx[ic]
            cryst.add_shape(legume.Circle(x_cent=xc, y_cent=yc, r=rads[ic] * self.radius))

            # Bounds here avoid double adding holes at edges of quadrants.
            if nx - 0.6 > self.xp[ic] > 0 and (ny - 1.1) * np.sqrt(3) / 2 > self.yp[ic] > 0:
                cryst.add_shape(legume.Circle(x_cent=-xc, y_cent=-yc, r=rads[ic] * self.radius))
            if nx - 1.6 > self.xp[ic] > 0:
                cryst.add_shape(legume.Circle(x_cent=-xc, y_cent=yc, r=rads[ic] * self.radius))
            if (ny - 1.1) * np.sqrt(3) / 2 > self.yp[ic] > 0 and nx - 1.1 > self.xp[ic]:
                cryst.add_shape(legume.Circle(x_cent=xc, y_cent=-yc, r=rads[ic] * self.radius))

        # et voila! the crystal should be defined.
        return cryst

    def get_supercell(self) -> Tuple[int, int]:
        """
        The number of periods in X and Y of base crystal in the supercell_size.
        @return: length 2 array of periods (Nx,Ny)
        """
        return self._supercell_size

    def get_num_holes(self) -> int:
        """
        @return: Number of independently varying holes of crystal. All inputs to cavity method should be of this form.
        """
        return self._num_holes

    def get_base_crystal(self) -> legume.phc.phc:
        """
        Generates the base crystal. The crystal if there was no defect to translational symmetry of holes.
        @return: photonic crystal object representing the base crystal.
        """

        # All supported topologies are currently hexagonal.

        lattice = legume.Lattice('hexagonal')
        cryst = legume.PhotCryst(lattice)
        cryst.add_layer(d=self.thickness, eps_b=self.eps)
        cryst.add_shape(legume.Circle(x_cent=0, y_cent=0, r=self.radius))
        return cryst
