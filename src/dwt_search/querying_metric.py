from heapq import nsmallest

from scipy.misc import imresize
from sklearn.pipeline import make_pipeline

from dwt_search import WaveletCoefficientTransformer, ThresholdTransformer


class ImageQueryingMetric:

    def __init__(self,
                 coeff_transformer: WaveletCoefficientTransformer,
                 thresholder: ThresholdTransformer,
                 in_shape,
                 weights,
                 color=True):
        self._coeff_extractor = make_pipeline(coeff_transformer, thresholder)
        self.shape = in_shape
        expected_2d_shape = tuple(d // 2 ** coeff_transformer.level for d in in_shape[:2])
        self.weights = validated_weights(weights, expected_2d_shape, color)

    def fit(self, imgs):
        """Transform imgs to appropriate coefficients and store them"""
        resized_imgs = [self.resize(img) for img in imgs]
        coeffs = self._coeff_extractor.transform(resized_imgs)
        self.data = coeffs

    def predict(self, imgs, n):
        """predict_single for each image in imgs"""
        return [self.predict_single(img, n) for img in imgs]

    def predict_single(self, img, n):
        """Return indices of n closest data points"""
        resized_img = self.resize(img)
        scores = self.__get_distances(resized_img)
        n_closest_tuples = nsmallest(n, enumerate(scores), key=lambda p: p[1])
        return list(zip(*n_closest_tuples))

    def metric(self, c1, c2):
        """Weighted metric from paper, falls back to l1 norm"""
        weighted_difference = self.weights * (c1 - c2)
        return self.default_norm(weighted_difference)

    def __get_distances(self, img):
        coeffs = self._coeff_extractor.transform([img])[0]
        return [self.metric(coeffs, datapoint) for datapoint in self.data]

    def resize(self, img):
        return imresize(img, self.shape)

    def default_norm(self, v):
        """l1 norm"""
        return abs(v).sum()


def validated_weights(weights, expected_shape, color):
    if color and weights.ndim != 3 or not color and weights.ndim != 2:
        raise AttributeError("Weights are of inappropriate shape")
    if weights.shape[:2] != expected_shape:
        raise AttributeError("Weights are of inappropriate shape")
    else:
        return weights