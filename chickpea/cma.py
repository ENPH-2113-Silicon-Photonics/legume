import numpy as np
import legume


class CavityModeAnalysis:

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

    def mode_volume(self, gme, field, components, kind, mind, sample_scale=2):
        """
        Get the Max
        """
        s_nx = int(sample_scale * self.nx)
        s_ny = int(sample_scale * self.ny)

        fields, _, _ = gme.get_field_xy(field=field, kind=kind, mind=mind, z=self.dslab / 2,
                                        component=components, Nx=s_nx, Ny=s_ny)

        field = np.zeros((s_ny, s_nx))
        for component in components:
            field = field + np.abs(fields[component]) ** 2

        field = field ** (1 / 2)

        defect_field = np.abs(field[int(self.marg_y * s_ny): -int(self.marg_y * s_ny),
                              int(self.marg_x * s_nx): -int(self.marg_x * s_nx)])

        mode_volume = 1 / np.square(np.amax(defect_field))

        return mode_volume

    def set_arglist(self, arg_list):
        self.arg_list = arg_list

    def mode_volume_filter(self, max_volume, band_filter=True, field='h', components='z'):
        """
        Finds the band gaps
        """

        if len(self.gme.freqs) == 0:
            raise ValueError("The guided mode expansion has no saved eigenvectors please run again.")

        arg_list = []

        shape = self.gme.freqs.shape
        for kind in range(shape[0]):
            arg_list.append(np.array(range(shape[1])))

        if band_filter is True:
            band_gaps, k_air, k_di = self.find_band_gaps()
            filt = np.zeros(shape, dtype=bool)
            for band_gap in band_gaps:
                filt = filt + (self.gme.freqs < (band_gap[1])) * (self.gme.freqs > band_gap[0])

            for kind, mlist in enumerate(arg_list):
                arg_list[kind] = mlist[filt[kind]]

        v = []
        for kind, mlist in enumerate(arg_list):
            vlist = []
            for mind in mlist:
                vlist.append(self.mode_volume(self.gme, field, components, kind, mind, sample_scale=2))
            vlist = np.array(vlist)

            # Filter out high mode volume modes.
            arg_list[kind] = mlist[vlist <= max_volume]
            vlist = vlist[vlist <= max_volume]
            v.append(vlist)

        return arg_list, v

    def categorize_cavity_modes(self, inds):
        """

        """

    def find_band_gaps(self, sample_rate=10, order=np.array([0, 3]), band_tol=0.1, lc_trim=0.5, numeig=10):
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
                              compute_im=False,
                              gradients='approx',
                              verbose=False)

        k_abs = np.tile((self.base_gme.kpoints[0] ** 2 + self.base_gme.kpoints[1] ** 2) ** (1 / 2), (numeig, 1)).T
        in_lc_freqs = self.base_gme.freqs[self.base_gme.freqs / (np.abs(k_abs - lc_trim) + 1e-10) <= 1 / (2 * np.pi)]

        freqs_flat = np.sort(in_lc_freqs)

        gaps = np.diff(freqs_flat)
        band_gaps = []
        for i in range(gaps.size):
            if gaps[i] >= band_tol:
                band_gaps.append([freqs_flat[i], freqs_flat[i + 1], (freqs_flat[i] + freqs_flat[i + 1]) / 2])

        band_gaps = np.array(band_gaps)

        if band_gaps.size == 0:
            return [], [], []

        k_air_arg = np.array([np.argwhere(self.base_gme.freqs == band_gap[1])[0][0] for band_gap in band_gaps])
        k_di_arg = np.array([np.argwhere(self.base_gme.freqs == band_gap[0])[0][0] for band_gap in band_gaps])

        k_air = (self.base_gme.kpoints[0][k_air_arg], self.base_gme.kpoints[1][k_air_arg])
        k_di = (self.base_gme.kpoints[0][k_di_arg], self.base_gme.kpoints[1][k_di_arg])

        return band_gaps, k_air, k_di
