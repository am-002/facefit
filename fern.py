import numpy as np
from util import rand_unit_vector

class FernBuilder:
    def __init__(self, n_pixels, n_features, n_landmarks, beta):
        self.n_pixels = n_pixels
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.beta = beta

    def _highest_correlated_pixel(self, dir, targets, cov_pp, pixel_values, pixel_averages, var_pp_sum):
        # # Project each target onto random direction.
        # lengths = targets.reshape((len(targets), 2*self.n_landmarks)).dot(dir)
        # cov_l_p = pixel_values.dot(lengths)/len(targets) - np.average(lengths) * pixel_averages
        # correlation = (cov_l_p[:, None] - cov_l_p) / np.sqrt(np.std(lengths) * (var_pp_sum - 2 * cov_pp))
        #
        # if np.isnan(correlation).all():
        #     return 0, 0
        # res = np.nanargmax(correlation)
        # return res / self.n_pixels, res % self.n_pixels

        # Random projection vector
        projection = np.random.normal(size=136)
        # Project regression targets against random projection
        Y_proj = targets.reshape(len(targets), 136).dot(projection)
        # Calculate covariance of projections
        Y_proj_cov = np.cov(Y_proj, bias=1)
        # Calculate the pixel covariance
        Y_pixels_cov  = (((pixel_values * Y_proj).sum(axis=1) / len(targets)) -
                         (Y_proj.sum() / len(targets)) * (pixel_values.sum(axis=1) / len(targets)))
        # Calculate the indices of the largest covariance
        pixel_cov_diff = np.diag(cov_pp)[:, None] + np.diag(cov_pp)
        correlation = (Y_pixels_cov[:, None] - Y_pixels_cov) / np.sqrt(
            Y_proj_cov * (pixel_cov_diff - 2 * cov_pp))
        res = np.nanargmax(correlation)

        return res/400, res%400

    @staticmethod
    def _get_features(pixel_samples, feature_indices):
        return pixel_samples[:, feature_indices[:, 0]] - pixel_samples[:, feature_indices[:, 1]]

    @staticmethod
    def _get_bin_ids(features, thresholds):
        return np.apply_along_axis(Fern.get_bin, arr=features, axis=1, thresholds=thresholds)

    @staticmethod
    def _calc_bin_averages(targets, bin_ids, n_features, n_landmarks, beta):
        bins = np.zeros((2 ** n_features, 2 * n_landmarks))
        bins_size = np.zeros((2 ** n_features,))
        bins[bin_ids] += targets.reshape(len(targets), 2 * n_landmarks)
        bins_size[bin_ids] += 1
        denoms = (bins_size + beta)
        # Replace all 0 denominators with 1 to avoid division by 0.
        denoms[denoms == 0] = 1
        return bins / denoms[:, None]

    def build(self, pixel_samples, targets, cov_pp, pixel_values, pixel_averages, pixel_var_sum):
        feature_indices = np.zeros((self.n_features, 2), dtype=int, order='c')

        for f in xrange(self.n_features):
            dir = rand_unit_vector(2*self.n_landmarks)
            feature_indices[f] = self._highest_correlated_pixel(dir, targets, cov_pp, pixel_values, pixel_averages, pixel_var_sum)

        features = self._get_features(pixel_samples, feature_indices)
        ranges = features.min(axis=0), features.max(axis=0)
        # thresholds = np.random.uniform(low=ranges[0], high=ranges[1])
        thresholds = (ranges[0]+ranges[1])/2.0 + np.random.uniform(low=-(ranges[1]-ranges[0])*0.1,
                                                                    high=(ranges[1]-ranges[0])*0.1)

        bin_ids = self._get_bin_ids(features, thresholds)
        bin_outputs = self._calc_bin_averages(targets, bin_ids, self.n_features, self.n_landmarks, self.beta)

        return Fern(self.n_landmarks, feature_indices, bin_outputs, thresholds)

class Fern:
    r"""
    Implementation of a random Fern.

    """
    def __init__(self, n_landmarks, feature_indices, bins, thresholds):
        self.n_landmarks = n_landmarks
        self.bins = bins
        self.features = feature_indices
        self.thresholds = thresholds

    @staticmethod
    def get_bin(features, thresholds):
        res = 0
        for i in xrange(len(features)):
            if features[i] <= thresholds[i]:
                res |= 1 << i
        return res

    def apply(self, pixels):
        # Select features from shape-indexed pixels.
        features = pixels[self.features[:, 0]]-pixels[self.features[:, 1]]
        # Get bin of the sample with the given shape-indexed pixels.
        bin_id = self.get_bin(features, self.thresholds)
        return self.bins[bin_id].reshape((self.n_landmarks,2))
