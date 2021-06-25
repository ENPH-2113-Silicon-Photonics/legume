import numpy as np


class XYFieldVizualizer:

    def __init__(self, gme, index_list, res, z_dimension, cell_shape, polarization, unfolded_disp=None, ):

        self.z = z_dimension
        self.cell_shape = cell_shape
        self.polarization = polarization

        h_comp, e_comp = {"TE": ['z', 'xy'],
                          "TM": ['xy', 'z'],
                          "None": ['xyz', 'xyz']
                          }[polarization]

        self.index_list = index_list

        if isinstance(res, int):
            self.res = (res, res)

        elif isinstance(res, tuple) and len(res) == 2:
            self.res = res

        fields = []

        if unfolded_disp is not None:
            self.dispersion = unfolded_disp
        else:
            self.dispersion = []

        for kind, mind in index_list:
            fe, _, _ = gme.get_field_xy(field='e', component=e_comp, z=z_dimension, kind=kind, mind=mind,
                                        Nx=self.res[0], Ny=self.res[1])
            fh, _, _ = gme.get_field_xy(field='h', component=h_comp, z=z_dimension, kind=kind, mind=mind,
                                        Nx=self.res[0], Ny=self.res[1])

            if polarization == 'TE':
                E = np.array([fe['x'], fe['y'], np.zeros(self.res)])
                H = np.array([np.zeros(self.res),np.zeros(self.res), fe['z']])
            elif polarization == 'TM':
                H = np.array([fe['x'], fe['y'], np.zeros(self.res)])
                E = np.array([np.zeros(self.res),np.zeros(self.res), fe['z']])
            elif polarization == 'None':
                E = np.array([fe['x'], fe['y'], fe['z']])
                H = np.array([fe['x'], fe['y'], fe['z']])

            fields.append((E,H))

            if unfolded_disp is None:
                self.dispersion.append((gme.kpoints[kind], gme.freqs[kind][mind]))

        xgrid, ygrid = (np.linspace(0, cell_shape[0], res), np.linspace(0, cell_shape[1], res))

        self.xgrid, self.ygrid = np.meshgrid(xgrid, ygrid)

        self.fields = fields
        self.basis = fields

        basis_rep = dict()
        for i in range(len(index_list)):
            basis_rep[i] = [i], [1]

        self.basis_rep = basis_rep

        self.poynting_vectors = []

    def set_basis(self, basis_rep):

        for index in basis_rep.keys():
            field_indices, factors = basis_rep(index)

            E_field = np.array([np.zeros(self.res), np.zeros(self.res), np.zeros(self.res)])

            H_field = np.array([np.zeros(self.res), np.zeros(self.res), np.zeros(self.res)])

            for i in range(len(field_indices)):
                E_field += self.fields[field_indices[i]][0]*factors[i]
                H_field += self.fields[field_indices[i]][1]*factors[i]

            self.basis[index] = (E_field, H_field)

        self.basis_rep = basis_rep

        # Reset basis dependent fields.
        self.poynting_vectors = []

    def modulate(self, phase, k):
        self.poynting_vectors = []

    def generate_poynting_vectors(self, time):
        poynting_vectors = []

        for field in self.basis:

            poynting_vector = np.cross(np.real(field[0]), np.real(field[1]), axis=0)
            poynting_vectors.append(poynting_vector)




