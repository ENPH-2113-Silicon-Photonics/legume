"""
Library of useful functions for chickpea
"""
import numpy as np

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