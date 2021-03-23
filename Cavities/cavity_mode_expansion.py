import numpy as np
import legume


class CavityModeExpansion:

    def __init__(self, phc, base_phc, super_periods, defect_margins, layer=0, gmax=3, base_gmax=4):

        self.base = base_phc
        self.layer = 0
        self.dslab = self.base.layers[layer].d

        self.nx = super_periods[0]
        self.ny = super_periods[1]

        self.marg_x = defect_margins[0]

        self.marg_y = defect_margins[1]

        self.arg_list = []
        self.gme = legume.GuidedModeExp(phc, gmax=gmax)
        self.base_gme = legume.GuidedModeExp(base_phc, gmax=base_gmax)

    def mode_volume(self, gme, sample_scale, kind, mind):
        """
        Get the Max
        """
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

    def set_q_factor(self):
        if self.freqs_im is not None:
            q_factor = []

            for kind, minds in self.arg_list:
                q_factor.append([kind, self.gme.freqs[kind][minds] / 2 / self.freqs_im[kind]])

            self.q_factor = np.array(q_factor)
        else:
            raise ValueError("Imaginary frequencies have not yet been found.")

    def find_band_gaps(self, sample_rate=10, order=np.array([0]), band_tol=0.1, lc_tol=0, numeig=20):
        lattice = self.base.lattice

        b1 = lattice.b1
        b2 = lattice.b2
        path = lattice.bz_path([[0, 0], b1 / 2, b2 / 2, [0, 0]],
                               [sample_rate * int(np.linalg.norm(b1) / np.pi),
                                sample_rate * int(np.linalg.norm(b1 - b2) / np.pi),
                                sample_rate * int(np.linalg.norm(b1) / np.pi)])

        if len(self.base_gme.freqs) == 0:
            self.base_gme.run(kpoints=path['kpoints'],
                              gmode_inds=order,
                              numeig=numeig,
                              verbose=False)

        k_squared = np.tile((self.base_gme.kpoints[0]**2+self.base_gme.kpoints[1]**2)**(1/2), (numeig, 1)).T
        in_lc_freqs = self.base_gme.freqs[self.base_gme.freqs/(k_squared+1e-12) <= 1/(2*np.pi) + lc_tol]

        freqs_flat = np.sort(in_lc_freqs)

        gaps = np.diff(freqs_flat)
        band_gaps = []
        for i in range(gaps.size):
            if gaps[i] >= band_tol:
                band_gaps.append([freqs_flat[i], freqs_flat[i + 1], (freqs_flat[i] + freqs_flat[i + 1]) / 2])

        gmg_ratio = np.array([(band_gap[1] - band_gap[0]) / band_gap[2] for band_gap in band_gaps])
        band_gaps = np.array(band_gaps)

        return band_gaps, gmg_ratio
