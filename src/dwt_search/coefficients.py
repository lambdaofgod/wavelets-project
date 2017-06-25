import pywt
import numpy as np


class WaveletCoefficientTransformer:

    def __init__(self, ndim, wavelet, mode, level):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        if ndim not in [2,3]:
            raise NotImplementedError("Wavelet coefficients for {}d are unsupported".format(ndim))
        else:
            self.ndim = ndim
            if ndim == 2:
                self.__get_coefficients = get_ca_coefficients2
            else:
                self.__get_coefficients = get_ca_coefficients3

    def fit_transform(self, images):
        def transform_single(image):
            if image.ndim == self.ndim:
                return self.__get_coefficients(image, self.wavelet, self.mode, self.level)
            else:
                raise AttributeError("Tried to get {}d transform in {}d transformer".format(image.ndim, self.ndim))
        return [transform_single(image) for image in images]


def get_ca_coefficients3(multi_channel_image, wavelet, mode='periodization', level=1):
    """Get coefficients for multichannel image"""
    def flatten_last_axis(img):
        return img.reshape(img.shape[:-1:])
    images_per_channel = np.dsplit(multi_channel_image, 3)
    coefficients_per_channel = (
        [get_ca_coefficients2(flatten_last_axis(image), wavelet, mode, level)
            for image in images_per_channel])
    return np.dstack(coefficients_per_channel)


def get_ca_coefficients2(image, wavelet, mode='periodization', level=1):
    """Get cA coefficients for grayscale image"""
    coeff_array = pywt.wavedec2(image, wavelet, mode, level)
    return coeff_array[0]
