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


def get_poynting_vector(E, H):
    assert E.shape[0]==3
    assert H.shape[0]==3

    return np.cross(E,H, axis=0)


