"""
Library of useful functions for chickpea
"""
import numpy as np
from PIL import Image, ImagePalette
import scipy as sp
import legume


def lowpass_downsample(bitmap, factor):
    """

    :param bitmap: Bitmap np array for factor
    :param factor: will cut
    :return:
    """
    double_factor = factor * 2

    fft_eps = np.fft.fft2(bitmap) / (bitmap.shape[0] * bitmap.shape[1])
    X = int(fft_eps.shape[0] / double_factor)
    Y = int(fft_eps.shape[1] / double_factor)
    fft_trunc = fft_eps[0:X, 0:X]

    top = np.vstack([fft_eps[0:X, 0:Y], fft_eps[-X:, 0:Y]])
    bot = np.vstack([fft_eps[0:X, -Y:], fft_eps[-X:, -Y:]])
    dft_trunc = np.hstack([top, bot])

    bitmap_lowpass = np.real(np.fft.ifft2(dft_trunc) * dft_trunc.shape[0] * dft_trunc.shape[1])

    return bitmap_lowpass


def import_eps_image(img_path, eps_map, tol=5):
    """

    :param img_path: Path to image
    :param eps_map: List of eps values. Image should have this many colors within tolerance set by tol.

                    Darkest points of images will be converted to first eps in eps_map.
                    Brightest will be converted to last point in eps_map.
    :param tol: tolerance to deviations from a color.

    :return: np array of permitivity values
    """

    # %%
    img = Image.open(img_path)

    img = img.quantize(colors=len(eps_map), dither=Image.FLOYDSTEINBERG)
    img = img.convert('L')

    # %%
    img = np.asarray(img)
    shape = img.shape

    eps_map = np.sort(eps_map)

    eps_array = np.zeros(shape)
    for i, eps in enumerate(eps_map):
        mask = img > img.max() - tol
        eps_array = eps_array + eps * mask
        img = img * (1 - mask)

    return eps_array


def find_band_gaps(gme, order, sample_rate=10, band_tol=0.1, trim_lc=False, lc_trim=0, numeig=20):
    lattice = gme.phc.lattice

    bz = lattice.get_irreducible_brioullin_zone_vertices()

    path = lattice.bz_path(bz, [sample_rate] * (len(bz) - 1))

    gme.run(kpoints=path['kpoints'],
            gmode_inds=order,
            numeig=numeig,
            compute_im=False,
            gradients='approx',
            verbose=False)

    k_abs = np.tile((gme.kpoints[0] ** 2 + gme.kpoints[1] ** 2) ** (1 / 2), (numeig, 1)).T
    if trim_lc:
        in_lc_freqs = gme.freqs[
            gme.freqs / (np.abs(k_abs - lc_trim) + 1e-10) <= 1 / (2 * np.pi)]

        freqs_flat = np.sort(in_lc_freqs)
    else:
        freqs_flat = np.sort(gme.freqs.flatten())

    gaps = np.diff(freqs_flat)
    band_gaps = []
    for i in range(gaps.size):
        if gaps[i] >= band_tol:
            band_gaps.append([freqs_flat[i], freqs_flat[i + 1], (freqs_flat[i] + freqs_flat[i + 1]) / 2])

    band_gaps = np.array(band_gaps)

    if band_gaps.size == 0:
        return [], [], []

    k_air_arg = np.array([np.argwhere(gme.freqs == band_gap[1])[0][0] for band_gap in band_gaps])
    k_di_arg = np.array([np.argwhere(gme.freqs == band_gap[0])[0][0] for band_gap in band_gaps])

    k_air = (gme.kpoints[0][k_air_arg], gme.kpoints[1][k_air_arg])
    k_di = (gme.kpoints[0][k_di_arg], gme.kpoints[1][k_di_arg])

    return band_gaps, k_air, k_di


def unfold_bands(super_gme, prim_gme, branch_start=-np.pi):
    # Collect g-vectors of primitive and supercell expansions.
    prim_gvecs = prim_gme.gvec.reshape(2, prim_gme.n1g, prim_gme.n2g)

    gvecs = super_gme.gvec.reshape(2, super_gme.n1g, super_gme.n2g)

    # TODO: We should try generating this other_mask in more analytic way.
    #   IE. Mask is periodic, we just need to find starting point. (0,0)
    #       New input would be supercell size (periodicity) and super_gme, we don't need prim_gme.

    mask = np.logical_and(np.isin(gvecs[0], prim_gvecs[0]), np.isin(gvecs[1], prim_gvecs[1]))

    # Determine periodicity of the crystal.

    lattice = super_gme.phc.lattice
    base_lattice = prim_gme.phc.lattice

    b1 = lattice.b1
    b2 = lattice.b2
    prim_b1 = base_lattice.b1
    prim_b2 = base_lattice.b2

    Nx = np.int_(np.rint(prim_b1 / b1))[0]
    Ny = np.int_(np.rint(prim_b2 / b2))[1]

    # Pad the other_mask so that when we roll the other_mask we don't create artifacts bleed array.
    # TODO If we generate other_mask from periodicity can we simply create shifted-period other_mask.

    pad_mask = np.pad(mask, ((Nx, 0), (Ny, 0)))

    dispersion = [[], [], []]
    probabilities = []

    for k in range(len(super_gme.kpoints.transpose())):
        for w in range(len(super_gme.eigvecs[k].transpose())):
            probability = []
            for i in range(Nx):
                probability.append([])
                for j in range(Ny):
                    trans_vec = (i, j)

                    # Roll padded other_mask and truncate to shape of eigen vector.

                    trans_mask = np.roll(pad_mask, trans_vec, axis=(0, 1))[Nx:, Ny:]

                    eig = super_gme.eigvecs[k].transpose()[w].reshape((super_gme.n1g, super_gme.n2g))
                    prob = np.sum(np.square(np.abs(eig[trans_mask])))

                    probability[-1].append(prob)

            probability = np.asarray(probability)

            # Normalize probability
            probability = probability / np.sum(probability)

            args = np.argwhere(probability == probability.max())[0]
            trans = (args[0]) * b1 + (args[1]) * b2

            new_k = super_gme.kpoints.T[k] + trans
            # Shift K to appropriate branch of mod function.
            new_k = np.mod(new_k - branch_start, 2 * np.pi) + branch_start

            dispersion[0].append(super_gme.freqs[k][w])
            dispersion[1].append(new_k[0])
            dispersion[2].append([k, w])
            probabilities.append(probability)

    return np.array(dispersion), np.array(probabilities)


def fixed_point_cluster(mean, diff, N, kernel='gaussian', u=None, reflect=False):
    """
    Generates a cluster of points.
    :param mean: Mean to generate points around
    :param diff: difference from mean. Distribution is anchored at these points.
    :param N: Number of points to generate.
    :param kernel: Function governing cluster. Should be non-negative and continuous.
                   If callable object given runs that function.
                   If string given looks up function from preset table.
    :param u: single parameter governing preset kernels
    :param reflect: Reflect the cluster around the zero axis.
    :return: cluster of points.
    """

    kern_dict = {"binomial": lambda x: np.abs(x + (u - 1) / (3 * diff ** 2) * x ** 3),
                 "id": lambda x: np.abs(x)}

    if isinstance(kernel, str):
        kernel = kern_dict[kernel]
    elif callable(kernel):
        kernel = kernel
    else:
        raise ValueError("Kernel not callable or string.")

    lin_array = np.linspace(mean - diff, mean + diff, N)
    clustered_array = np.sign(lin_array - mean) * diff * kernel(lin_array - mean) / kernel(diff) + mean

    if reflect:
        clustered_array = np.concatenate((-clustered_array, clustered_array))

    return clustered_array
