import numpy as np
import legume





class CavityModeExpansion:

    def __init__(self, base_crystal, super_crystal, super_periods,
                 polarization, defect_margins=(0.25, 0.25), layer=0, gmax=3, base_gmax=4):

        self.base = base_crystal
        self.layer = 0
        self.dslab = self.base.layers[layer].d

        self.nx = super_periods[0]
        self.ny = super_periods[1]

        self.marg_x = defect_margins[0]

        self.marg_y = defect_margins[1]
        self.pole = polarization

        self.freqs_im = None
        self.rad_coup = None
        self.rad_gvec = None
        self.q_factor = None
        self.band_gaps = None
        self.arg_list = []
        self.gme = legume.GuidedModeExp(super_crystal, gmax=gmax)
        self.base_gme = legume.GuidedModeExp(base_crystal, gmax=base_gmax)

    def mode_volume(self, gme, sample_scale, kind, mind):
        # Get the electric field in the center of the slab
        s_nx = int(sample_scale * self.nx)
        s_ny = int(sample_scale * self.ny)

        field = gme.get_field_xy(self.pole, kind=kind, mind=mind, z=self.dslab / 2,
                                 component='z', Nx=s_nx, Ny=s_ny)[0]['z']

        defect_field = np.abs(field[int(self.marg_x * s_nx): -int(-self.marg_x * s_nx),
                              int(self.marg_y * s_ny): -int(self.marg_y * s_ny)])

        mode_volume = 1 / np.square(np.amax(defect_field))

        leakage = 1 - np.sum(defect_field)

        return mode_volume, leakage

    def set_arglist(self, arg_list):
        self.arg_list = arg_list

    def get_im_freq(self):
        freqs_im = []
        rad_coup = []
        rad_gvec = []
        for kind, minds in self.arg_list:
            if self.gme.freqs_im.size() == 0:
                freqs_im, rad_coup, rad_gvec = self.gme.compute_rad(kind=kind, minds=minds)
                freqs_im.append([kind, freqs_im])
                rad_coup.append([kind, rad_coup])
                rad_gvec.append([kind, rad_gvec])
            else:
                freqs_im.append([kind, self.gme.freqs_im[kind][minds]])
                rad_coup.append([kind, self.gme.rad_coup[kind][minds]])
                rad_gvec.append([kind, self.gme.rad_gvec[kind][minds]])

        self.freqs_im = np.array(freqs_im)
        self.rad_coup = np.array(rad_coup)
        self.rad_gvec = np.array(rad_gvec)

    def set_q_factor(self):
        if self.freqs_im is not None:
            q_factor = []

            for kind, minds in self.arg_list:
                q_factor.append([kind, self.gme.freqs[kind][minds] / 2 / self.freqs_im[kind]])

            self.q_factor = np.array(q_factor)
        else:
            raise ValueError("Imaginary frequencies have not yet been found.")

    def find_band_gaps(self, sample_rate=10, order=np.array([0]), band_tol=0.1):
        lattice = self.base.lattice

        b1 = lattice.b1
        b2 = lattice.b2
        path = lattice.bz_path([[0, 0], b1 / 2, b2 / 2, [0, 0]],
                               [sample_rate * int(np.linalg.norm(b1) / np.pi),
                                sample_rate * int(np.linalg.norm(b1 - b2) / np.pi),
                                sample_rate * int(np.linalg.norm(b1) / np.pi)])

        if self.base_gme.freqs.size() == 0:
            self.base_gme.run(kpoints=path['kpoints'],
                              gmode_inds=order + self.pole,
                              numeig=20,
                              verbose=False)

        freqs = np.sort(self.base_gme.freqs.flatten())
        gaps = np.argsort(np.diff(np.sort(self.base_gme.freqs.flatten())))
        band_gaps = []

        i = 0
        while gaps[i] >= band_tol:
            band_gaps.append([freqs[gaps[i]], freqs[gaps[i]+1], (freqs[gaps[i]]+freqs[gaps[i]+1])/2])

        self.band_gaps = band_gaps

        return band_gaps

