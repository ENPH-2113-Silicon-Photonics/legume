import numpy as np
from legume.backend import backend as bd
import legume.utils as utils
import itertools
from scipy.spatial import Voronoi


class Lattice(object):
    """
    Class for constructing a Bravais lattice
    """

    def __init__(self, *args, reduce_lattice=True):
        """
        Initialize a Bravais lattice.
        If a single argument is passed, then

            - 'square': initializes a square lattice.
            - 'hexagonal': initializes a hexagonal lattice.

        with lattice constant a = 1 in both cases.

        If two arguments are passed, they should each be 2-element arrays
        defining the elementary vectors of the lattice.
        """

        # Primitive vectors cell definition
        (a1, a2) = self._parse_input(*args, reduce_lattice=reduce_lattice)
        self.reduced = reduce_lattice

        self.a1 = a1[0:2]
        self.a2 = a2[0:2]

        ec_area = bd.norm(bd.cross(a1, a2))
        a3 = bd.array([0, 0, 1])

        # Reciprocal lattice basis vectors
        b1 = 2 * np.pi * bd.cross(a2, a3) / bd.dot(a1, bd.cross(a2, a3))
        b2 = 2 * np.pi * bd.cross(a3, a1) / bd.dot(a2, bd.cross(a3, a1))

        bz_area = bd.norm(bd.cross(b1, b2))

        self.b1 = b1[0:2]
        self.b2 = b2[0:2]

        self.ec_area = ec_area  # Elementary cell area
        self.bz_area = bz_area  # Brillouin zone area

    def __repr__(self):
        return "Lattice(a1 = [%.4f, %.4f], a2 = [%.4f, %.4f])" % \
               (self.a1[0], self.a1[1], self.a2[0], self.a2[1])

    def _get_recip_voronoi(self):
        points = []

        for i, j in itertools.combinations([-1, 0, 1], 2):
            points.append(i * self.b1 + j * self.b2)

        return Voronoi(points)

    def get_irreducible_brioullin_zone_vertices(self):
        if not self.reduced:
            raise UserWarning("Lattice not reduced, algorithm depends on reduced lattice to function.")

        if self.type == 'square':
            return ['G', self.b1 / 2, self.b1 / 2 + self.b2 / 2, 'G']
        elif self.type == 'custom_square':
            return ['G', self.b1 / 2, self.b1 / 2 + self.b2 / 2, 'G']
        elif self.type == 'hexagonal':
            return ['G', self.b1 / 2, 1 / np.linalg.norm(self.a1) * np.array([4 / 3 * np.pi, 0]), 'G']
        elif self.type == 'custom_hexagonal':
            return ['G', self.b1 / 2, 1 / np.linalg.norm(self.a1) * np.array([4 / 3 * np.pi, 0]), 'G']
        elif self.type == 'rectangular':
            return ['G', self.b1 / 2, self.b1 / 2 + self.b2 / 2, self.b2 / 2, 'G']
        elif self.type == 'custom':
            raise NotImplementedError()
        pass

    def _parse_input(self, *args, reduce_lattice):

        if len(args) == 1:
            if args[0] == 'square':
                self.type = 'square'
                a1 = bd.array([1, 0, 0])
                a2 = bd.array([0, 1, 0])
            elif args[0] == 'hexagonal':
                self.type = 'hexagonal'
                a1 = bd.array([0.5, bd.sqrt(3) / 2, 0])
                a2 = bd.array([0.5, -bd.sqrt(3) / 2, 0])
            else:
                raise ValueError("Lattice can be 'square' or 'hexagonal, "
                                 "or defined through two primitive vectors.")

        elif len(args) == 2:
            a1 = bd.hstack((bd.array(args[0]), 0))
            a2 = bd.hstack((bd.array(args[1]), 0))
            if reduce_lattice:
                a1, a2 = self._gauss_reduction(a1[0:2], a2[0:2])
                a1 = bd.hstack((a1, 0))
                a2 = bd.hstack((a2, 0))
            if np.round(np.abs(bd.dot(a1, a2) / bd.dot(a1, a1)), 6) == 0.5 \
                    and np.round(bd.norm(a1), 6) == np.round(bd.norm(a2), 6):

                self.type = 'custom_hexagonal'
            elif bd.dot(a1, a2) == 0:
                if bd.norm(a1) == bd.norm(a2):
                    self.type = 'custom_square'
                else:
                    self.type = 'rectangular'
            else:
                self.type = 'custom'

        return a1, a2

    @staticmethod
    def _gauss_reduction(v1, v2):
        """
        Lenstra–Lenstra–Lovász lattice reduction
        :param v1: Lattice Vector 1
        :param v2: Lattice Vector 2
        :return: a1, a2: Reduced Lattice Vectors
        """
        # See https://kel.bz/post/lll/
        a1,a2=v1,v2
        while True:
            n1 = bd.norm(v1)
            n2 = bd.norm(v2)
            if bd.norm(n2) < bd.norm(n1):
                v1, v2 = v2, v1  # swap step

            # We need 0.5 to round up so we use floor(0.5+x) as round.
            m = bd.floor(0.5 + bd.dot(v1, v2) / bd.dot(v1, v1))
            if m <= 0:
                if bd.dot(a1, v2)**2+bd.dot(a2, v1)**2>bd.dot(a2, v2)**2+bd.dot(a1, v1)**2:
                    # Maximizes projection onto original basis.

                    v1,v2 = v2,v1
                return v1, v2
            v2 = v2 - m * v1  # reduction step

    def xy_grid(self, Nx=100, Ny=100, periods=None):
        """
        Define an xy-grid for visualization purposes based on the lattice
        vectors.
        
        Parameters
        ----------
        Nx : int, optional
            Number of points along `x`.
        Ny : int, optional
            Number of points along `y`.
        periods : float, optional
            A number or a list of two numbers that defines how many periods 
            in the `x`- and `y`-directions are included. 
        
        Returns
        -------
        np.ndarray
            Two arrays defining a linear grid in `x` and `y`.
        """
        if periods == None:
            periods = [1, 1]
        elif np.array(periods).shape == 1:
            periods = periods[0] * np.ones((2,))

        ymax = np.abs(max([self.a1[1], self.a2[1]])) * periods[1] / 2
        ymin = -ymax

        xmax = np.abs(max([self.a1[0], self.a2[0]])) * periods[0] / 2
        xmin = -xmax

        return (np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))

    def bz_path(self, pts, ns):
        """
        Make a path in the Brillouin zone.
        
        Parameters
        ----------
        pts : list
            A list of points. Each element can be either a 2-element array 
            defining (kx, ky), or one of {'G', 'K', 'M'} for a 'hexagonal' 
            Lattice type, or one of {'G', 'X', 'M'} for a 'square' Lattice 
            type. 
        ns : int or list
            A list of length either 1 or ``len(pts) - 1``, specifying 
            how many points are to be added between each two **pts**.
        
        Returns
        -------
        path: dict 
            A dictionary with the 'kpoints', 'labels', and the 
            'indexes' corresponding to the labels.      
        """
        if isinstance(ns, int):
            ns = [ns]

        if not isinstance(ns, list):
            ns = list(ns)
        npts = len(pts)
        if npts < 2:
            raise ValueError("At least two points must be given")

        if len(ns) == 1:
            ns = ns[0] * np.ones(npts - 1, dtype=np.int_)
        elif len(ns) == npts - 1:
            ns = np.array(ns)
        else:
            raise ValueError("Length of ns must be either 1 or len(pts) - 1")

        kpoints = np.zeros((2, np.sum(ns) + 1))
        inds = [0]
        count = 0

        for ip in range(npts - 1):
            p1 = self._parse_point(pts[ip])
            p2 = self._parse_point(pts[ip + 1])
            kpoints[:, count:count + ns[ip]] = p1[:, np.newaxis] + np.outer( \
                (p2 - p1), np.linspace(0, 1, ns[ip], endpoint=False))
            count = count + ns[ip]
            inds.append(count)
        kpoints[:, -1] = p2

        path = {
            'kpoints': kpoints,
            'labels': [str(pt) for pt in pts],
            'indexes': inds
        }

        return path

    def _parse_point(self, pt):
        """
        Returns a numpy array corresponding to a BZ point pt
        """
        if type(pt) == np.ndarray:
            return pt
        elif type(pt) == str:
            if pt.lower() == 'g' or pt.lower() == 'gamma':
                return np.array([0, 0])

            if pt.lower() == 'x':
                if self.type == 'square':
                    return np.array([np.pi, 0])
                else:
                    raise ValueError("'X'-point is only defined for lattice "
                                     "initialized as 'square'.")

            if pt.lower() == 'm':
                if self.type == 'square':
                    return np.array([np.pi, np.pi])
                elif self.type == 'hexagonal':
                    return np.array([np.pi, np.pi / np.sqrt(3)])
                else:
                    raise ValueError("'М'-point is only defined for lattice "
                                     "initialized as 'square' or 'hexagonal'.")

            if pt.lower() == 'k':
                if self.type == 'hexagonal':
                    return np.array([4 / 3 * np.pi, 0])
                else:
                    raise ValueError("'K'-point is only defined for lattice "
                                     "initialized as 'hexagonal'.")

        raise ValueError("Something was wrong with BZ point definition")
