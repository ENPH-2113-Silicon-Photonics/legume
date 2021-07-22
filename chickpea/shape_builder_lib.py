import numpy as np

class ShapeBuilder():
    def __init__(self, parameter_keys, bounds, **defaults):

        self._parameters = parameter_keys
        self._defaults = dict()

        self._defaults.update(defaults)
        self._bounds = bounds

        if not self._bounds.keys()==self._parameters:
            raise ValueError("Bounds namespace not same as parameter key list.")
        if not self.defaults.keys().issubset(self._parameters):
            raise ValueError("Default keys not in  namespace not same as parameter key list.")


    def place_shape(self, phc, eps, x, y, **params):
        """
        Takes in a PhotCryst object and places shapes in it.
        :param phc: PhotCryst crystal object to place shapes in.
        :param params: parameters governing shape
        :return:
        """
        raise NotImplementedError("Must be implemented by subclass")

    @property
    def bounds(self):
        """
        Get bounds on parameters
        :return: returns dictionary of bounds on parameters.
        """
        return self._bounds
    @bounds.setter
    def bounds(self, bounds):
        """
        Set bounds on parameters
        :return: returns dictionary of bounds on parameters.
        """
        self._bounds = bounds

    @property
    def defaults(self):
        """
        Returns default parameters
        :return: returns dictionary of default parameters.
        """
        return self._defaults

    @defaults.setter
    def defaults(self, **defaults):
        """
        Sets default parameters
        """
        for key in defaults.keys():
            if not self._bounds[key][0] < default[key] < self._bounds[key][1]:
                raise ValueError("Default parameter %s out of bounds" % key)

        self._defaults = defaults

    @property
    def parameters(self):
        """
        Returns keys of parameters of shapes.
        :return:
        """
        return self._parameters

class c6v_trihex(ShapeBuilder):
    """
        Six triangles in c6v symmetry.
    """
    def __init__(self, eps, L=0.75, base=np.tan(np.pi/3), height=0.5, a=1):

        self.default_params={"eps": eps,
                             "L": L,
                             "base": base,
                             "height": height
                             }
        self.a = a

        raise NotImplementedError("Must be implemented by subclass")


    def place_shape(self, phc, **params):

        params_ = self.default_params.copy()
        params_.update(params)

        x = params_['x']
        y = params_['y']
        eps = params_['eps']
        L=params_['l']
        base=params_['base']
        height=params_['height']

        v1 = np.array([L-height,0])
        v2 = np.array([L,base/2])
        v3 = np.array([L,-base/2])
        V = np.hstack(v1.T,v2.T,v3.T)

        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        for i in range(6):
            rotation=np.linalg.matrix_power(R, i)
            rot_vectors=np.matmul(R,V)
            poly = legume.Poly(eps=eps, x_edges=rot_vectors[0]+x, y_edges=rot_vectors[1]+x)
            phc.add_shape(poly)


class hex_c6v_poly(ShapeBuilder):

    def __init__(self, N_vertices, a, **defaults):
        """
        Initialize Shape Builder
        :param N_vertices:
        :param a:
        """
        parameters = ['lengths', 'angles']

        bounds = {'lengths': [[0,a]]*N_vertices,
                  'angles':  [[0,np.radians(30)]]*N_vertices}

        super(parameters, bounds, **defaults)
        self.N_vertices = N_vertices

        self.a = a


    def place_shape(self, phc, eps, x, y, **params):
        """
        Takes in a PhotCryst object and places shapes in it.
        :param phc: PhotCryst crystal object to place shapes in.
        :param params: parameters governing shape
        """

        lengths = params['lengths']
        angles = params['angles']

        vectors=[]
        for ver_ind in range(self.N_vertices):
            l, theta=lengths[ver_ind], theta=angles[ver_ind]

            vectors.append(np.array(l, l*np.tan(theta)).T)

        rot_ang = np.radians(60)
        c, s = np.cos(rot_ang), np.sin(rot_ang)
        rot_mat = np.array(((c, -s), (s, c)))
        flip_mat = np.array((1,0),(0,-1))

        vec_mat_ = np.hstack(vectors)
        vec_mat = np.hstack((vec_mat_,np.matmul(flip_mat, vec_mat_)))

        for rot_power in range(6):
            rotation = np.linalg.matrix_power(rot_mat, rot_power)

            rot_vectors = np.matmul(rotation, vec_mat)

            poly = legume.Poly(eps=eps, x_edges=rot_vectors[0] + x, y_edges=rot_vectors[1] + y)
            phc.add_shape(poly)

