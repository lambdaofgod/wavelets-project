from dwt_search import WaveletCoefficientTransformer, ThresholdTransformer
import numpy as np
from heapq import nsmallest
from scipy.misc import imresize
from sklearn.pipeline import make_pipeline


class ImageQueryingMetric:

    def __init__(self,
                 coeff_transformer: WaveletCoefficientTransformer,
                 thresholder: ThresholdTransformer,
                 in_shape,
                 weights,
                 color=True):
        self._coeff_extractor = make_pipeline(coeff_transformer, thresholder)
        self.shape = in_shape
        expected_2d_shape = [d // 2 ** coeff_transformer.level for d in in_shape[:2]]
        self.weights = validated_weights(weights, expected_2d_shape, color)

    def fit(self, imgs):
        resized_imgs = [self.resize(img) for img in imgs]
        coeffs = self._coeff_extractor.fit_transform(resized_imgs)
        self.data = coeffs

    def predict(self, imgs, n):
        return [self.predict_single(img, n) for img in imgs]

    def predict_single(self, img, n):
        resized_img = self.resize(img)
        scores = self.__get_distances(resized_img)
        n_largest_indices = nsmallest(n, enumerate(scores), key=lambda p: p[1])
        return n_largest_indices

    def metric(self, c1, c2, norm=default_norm):
        return norm(self.weights * (c1 - c2))

    def __get_distances(self, img):
        coeffs = self._coeff_extractor.fit_transform([img])[0]
        return [self.metric(coeffs, datapoint) for datapoint in self.data]

    def resize(self, img):
        return imresize(img, self.shape)


def default_norm(v):
    return np.linalg.norm(v, ord=1)


def validated_weights(weights, expected_shape, color):
    if color and weights.ndim != 3 or not color and weights.ndim != 2:
        raise AttributeError("Weights are of inappropriate shape")
    if weights.shape[:2] != expected_shape:
        raise AttributeError("Weights are of inappropriate shape")
    else:
        return weights