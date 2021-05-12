import numpy as np
from legume.backend import backend as bd
import legume.utils as utils
from .shapes import Shape, Circle, Poly, Square
import cv2 as cv

class Layer(object):
    """
    Class for a single layer in the potentially multi-layer PhC.
    """
    def __init__(self, lattice, z_min: float=0, z_max: float=0):
        """Initialize a Layer.
        
        Parameters
        ----------
        lattice : Lattice
            A lattice defining the 2D periodicity.
        z_min : float, optional
            z-coordinate of the bottom of the layer.
        z_max : float, optional
            z-coordinate of the top of the layer.
        """
        # Define beginning and end in z-direction
        self.z_min = z_min
        self.z_max = z_max

        # Slab thickness
        self.d = z_max - z_min

        # Effective permittivity
        self._eps_eff = None

        # Underlying lattice
        self.lattice = lattice

    def __repr__(self):
        return 'Layer'

    @property
    def eps_eff(self):
        if self._eps_eff is None:
            raise ValueError("Layer effective epsilon not set, use "
                                "`layer.eps_eff = ...` to set")
        else:
            return self._eps_eff

    @eps_eff.setter
    def eps_eff(self, eps):
        self._eps_eff = eps
    

    def compute_ft(self, gvec):
        """
        Compute the 2D Fourier transform of the layer permittivity.
        """
        raise NotImplementedError("compute_ft() needs to be implemented by"
            "Layer subclasses")

    def get_eps(self, points):
        """
        Compute the permittivity of the layer over a 'points' tuple containing
        a meshgrid in x, y defined by arrays of same shape.
        """
        raise NotImplementedError("get_eps() needs to be implemented by"
            "Layer subclasses")

class ShapesLayer(Layer):
    """
    Layer with permittivity defined by Shape objects
    """
    def __init__(self, lattice, z_min: float=0, z_max: float=0,
                    eps_b: float=1.):
        """Initialize a ShapesLayer.
        
        Parameters
        ----------
        lattice : Lattice
            A lattice defining the 2D periodicity.
        z_min : float, optional
            z-coordinate of the bottom of the layer.
        z_max : float, optional
            z-coordinate of the top of the layer.
        eps_b : float, optional
            Layer background permittivity.
        """
        super().__init__(lattice, z_min, z_max)

        # Define background permittivity
        self.eps_b = eps_b

        # Initialize average permittivity - needed for guided-mode computation
        self.eps_avg = bd.array(eps_b)

        # Initialize an empty list of shapes
        self.layer_type = 'shapes'
        self.shapes = []

    def __repr__(self):
        rep = 'ShapesLayer(eps_b = %.2f, d = %.2f' % (self.eps_b, self.d)
        rep += ',' if len(self.shapes) > 0 else ''
        for shape in self.shapes:
            rep += '\n' + repr(shape)
        rep += '\n)' if len(self.shapes) > 0 else ')'
        return rep

    def add_shape(self, shapes):
        """
        Add a shape or a list of shapes to the layer.
        """
        if isinstance(shapes, Shape):
            shapes = [shapes]

        for shape in shapes:
            if isinstance(shape, Shape):
                self.shapes.append(shape)
                self.eps_avg = self.eps_avg + (shape.eps - self.eps_b) * \
                                shape.area/self.lattice.ec_area
            else:
                raise ValueError("Argument to add_shape must only contain "
                "instances of legume.Shape (e.g legume.Circle or legume.Poly)")

    def compute_ft(self, gvec):
        """
        Compute the 2D Fourier transform of the layer permittivity.
        """

        FT = bd.zeros(gvec.shape[1])
        for shape in self.shapes:
            # Note: compute_ft() returns the FT of a function that is one 
            # inside the shape and zero outside
            FT = FT + (shape.eps - self.eps_b)*shape.compute_ft(gvec)

        # Apply some final coefficients
        # Note the hacky way to set the zero element so as to work with
        # 'autograd' backend
        ind0 = bd.abs(gvec[0, :]) + bd.abs(gvec[1, :]) < 1e-10  
        FT = FT / self.lattice.ec_area
        FT = FT*(1-ind0) + self.eps_avg*ind0

        return FT

    def get_eps(self, points):
        """
        Compute the permittivity of the layer over a 'points' tuple containing
        a meshgrid in x, y defined by arrays of same shape.
        """
        xmesh, ymesh = points
        if ymesh.shape != xmesh.shape:
            raise ValueError(
                    "xmesh and ymesh must have the same shape")

        eps_r = self.eps_b * bd.ones(xmesh.shape)

        # Slightly hacky way to include the periodicity
        a1 = self.lattice.a1
        a2 = self.lattice.a2

        a_p = min([np.linalg.norm(a1), 
                   np.linalg.norm(a2)])
        nmax = np.int_(np.sqrt(np.square(np.max(abs(xmesh))) + 
                        np.square(np.max(abs(ymesh))))/a_p) + 1

        for shape in self.shapes:
            for n1 in range(-nmax, nmax+1):
                for n2 in range(-nmax, nmax+1):
                    in_shape = shape.is_inside(xmesh + 
                        n1*a1[0] + n2*a2[0], ymesh + 
                        n1*a1[1] + n2*a2[1])
                    eps_r[in_shape] = utils.get_value(shape.eps)

        return eps_r


class FreeformLayer(Layer):
    """
    Layer with permittivity defined by a freeform distribution on a grid
    """
    def __init__(self, lattice, z_min=0, z_max=0, eps_dist=None, eps_b=1):
        super().__init__(lattice, z_min, z_max)


        # TODO: Is this a neccessary restriction? Is there a less strict restriciton?

        a1 = self.lattice.a1

        a2 = self.lattice.a2

        if bd.dot(a1, a2) != 0:
            raise ValueError("Only Rectangular Lattices for the FreeformLayer")

        self.eps_b=eps_b
        # Initialize average permittivity - needed for guided-mode computation
        if eps_dist is not None:
            self.res = eps_dist.shape

            self.eps_avg = np.sum(eps_dist)/(self.res[0]*self.res[1])

            self._eps_dist = eps_dist
            self._eps_ft = np.fft.fft2(eps_dist)
            self.initialized = True
        else:

            self.eps_avg = np.array(eps_b)

            self.res = None
            self._eps_dist = None
            self._eps_ft = None
            self.initialized = False
        # Initialize an empty list of shapes
        self.layer_type = 'freeform'

    def initialize(self, eps_dist):

        self.res = eps_dist.shape
        self.eps_avg = np.sum(eps_dist)/(self.res[0]*self.res[1])

        self._eps_dist = eps_dist
        self._eps_ft = np.fft.fft2(eps_dist)

        self.initialized = True

    def compute_ft(self, inds):
        if self.initialized:
            FT = []
            for (i_x, i_y) in inds.T:
                FT.append(self._eps_ft[i_x, i_y]*np.exp(-1j*np.pi*(i_x+i_y)))

            FT = np.array(FT, dtype=np.complex128)

        else:
            FT = bd.zeros(inds.shape[1])

        # Apply some final coefficients
        # Note the hacky way to set the zero element so as to work with
        # 'autograd' backend
        """
        Compute the 2D Fourier transform of the layer permittivity from the *indices* of the gvectors
        """
        ind0 = bd.abs(inds[0, :]) + bd.abs(inds[1, :]) < 1e-10
        FT = FT / (self.res[0]*self.res[1])
        FT = FT*(1-ind0) + self.eps_avg*ind0

        return FT

    def get_eps(self, points):
        """
        Compute the permittivity of the layer over a 'points' tuple containing
        a meshgrid in x, y defined by arrays of same shape.
        """
        xmesh, ymesh = points
        if self.initialized:

            inds_x = np.int_(xmesh * self.res[0] / np.linalg.norm(self.lattice.a1))
            inds_y = np.int_(ymesh * self.res[1] / np.linalg.norm(self.lattice.a2))
            return self._eps_dist[inds_x, inds_y]

        else:

            return self.eps_b * bd.ones(xmesh.shape)

