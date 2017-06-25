from heapq import nlargest
import numpy as np


class ThresholdTransformer:

    def __init__(self, n, binary=True):
        self.n = n
        self.binary = binary

    def fit_transform(self, X):
        return self.__thresholder(self.binary)(self.n, X)

    def __thresholder(self, binary):
        if binary:
            return threshold_smallest_sign
        else:
            return threshold_smallest


def threshold_smallest(n, data):
    """Truncate values in data ndarray that are less than n-th value"""
    n_largest_coeff = min(nlargest(n, data.flat))
    is_largest = abs(data) >= n_largest_coeff
    thresholded = data * is_largest
    return thresholded


def threshold_smallest_sign(n, data):
    """Truncate values and then get their sign"""
    return np.sign(threshold_smallest(n, data))