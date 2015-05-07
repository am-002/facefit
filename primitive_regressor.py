import random
import menpo
from menpo.shape import PointCloud
import numpy as np

INF = 102345678
BETA = 1000
# TODO !!!
#BETA = 0.0

class PrimitiveRegressor:
    def __init__(self, n_features, n_fern_features, n_ferns, n_landmarks):
        self.nFerns = n_ferns
        self.nFernFeatures = n_fern_features
        self.nFeatures = n_features
        self.nLandmarks = n_landmarks
        self.fern = Fern(n_fern_features, n_landmarks)
        self.fern_feature_selector = FeatureSelector(n_features, n_fern_features)

    def train(self, feature_vectors, targets, cov_pp, pixels, pixel_averages, pixel_var_sum):
        self.fern_feature_selector.train(targets, cov_pp, pixels, pixel_averages, pixel_var_sum)
        fern_feature_vectors = []
        for feature_vector in feature_vectors:
            fern_feature_vectors.append(self.fern_feature_selector.extract_features(feature_vector))
        self.fern.train(fern_feature_vectors, targets)

    def apply(self, shapeIndexedFeatures):
        fern_features = self.fern_feature_selector.extract_features(shapeIndexedFeatures)

        return self.fern.test(fern_features)


class Fern:
    r"""
    Implementation of a random Fern.

    """
    class Bin:
        def __init__(self, n_landmarks):
            self.size = 0.0
            self.delta_sum = menpo.shape.PointCloud([[0.0, 0.0]] * n_landmarks)

        def add(self, delta):
            self.size += 1.0
            self.delta_sum.points += delta.points

        def get_delta(self):
            if self.size == 0.0:
                return self.delta_sum
            return PointCloud(self.delta_sum.points / (self.size + BETA))


    def __init__(self, n_features, n_landmarks):
        self.n_features = n_features
        self.thresholds = [0] * self.n_features
        self.bins = [Fern.Bin(n_landmarks) for _ in range(2 ** n_features)]

    def get_bin(self, feature_vector):
        res = 0
        for i in range(self.n_features):
            if feature_vector[i] <= self.thresholds[i]:
                res |= 1 << i
        return res

    def train(self, feature_vectors, targets):
        n_samples = len(feature_vectors)

        # Generate thresholds for each features. We first get the ranges
        # of all features in the training set.
        ranges = [(INF, -INF)] * self.n_features

        for feature_vector in feature_vectors:
            for i, feature in enumerate(feature_vector):
                ranges[i] = (min(ranges[i][0], feature), max(ranges[i][1], feature))

        # Generate a random threshold for each feature.
        self.thresholds = [random.uniform(rng[0], rng[1]) for rng in ranges]

        for i in xrange(n_samples):
            bin_id = self.get_bin(feature_vectors[i])
            self.bins[bin_id].add(PointCloud(targets[i].reshape(68,2)))

    def test(self, feature_vector):
        bin_id = self.get_bin(feature_vector)
        return self.bins[bin_id].get_delta()


class FeatureSelector:
    def __init__(self, n_pixels, n_fern_features):
        self.n_pixels = n_pixels
        self.n_fern_features = n_fern_features
        self.features = np.zeros((2, n_fern_features), order='C', dtype=int)

    def train(self, targets, cov_pp, pixels, pixel_averages, pixel_var_sum):
        n_images = len(targets)
        n_landmarks = len(targets[0])/2

        for f in xrange(self.n_fern_features):
            # Project the offset in random direction and measure the scalar length
            # of the projection. Find which feature has highest imapct on the length.

            # Generate random direction.
            dir = np.random.randn(2*n_landmarks)

            # Normalize it.
            dir /= np.linalg.norm(dir)

            # Project each target onto random direction.
            lengths = targets.reshape((n_images, 2*n_landmarks)).dot(dir)

            cov_l_p = pixels.dot(lengths)/n_images - np.average(lengths) * pixel_averages

            # TODO: Should the following be lengths_std = np.cov(lengths, bias=1)?
            lengths_std = np.std(lengths)

            # Calculate the indices of the largest correlation
            #pixel_var_sum = np.diag(cov_pp)[:, None] + np.diag(cov_pp)
            correlation = (cov_l_p[:, None] - cov_l_p) / np.sqrt(
                lengths_std * (pixel_var_sum - 2 * cov_pp))
            res = np.nanargmax(correlation)

            self.features[0][f] = res/self.n_pixels
            self.features[1][f] = res%self.n_pixels

            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).
            # Here the features array is populated

    def extract_features(self, pixels):
        #return [pixels[p[0]] - pixels[p[1]] for p in self.features]
        return pixels[self.features[0]]-pixels[self.features[1]]
