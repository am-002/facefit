import numpy as np
from util import rand_unit_vector

import cv2

class FernBuilder:
    def __init__(self, n_pixels, n_features, n_landmarks, beta):
        self.n_pixels = n_pixels
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.beta = beta

    def _highest_correlated_pixel(self, dir, targets, cov_pp, pixel_values, pixel_averages, var_pp_sum):
        # # Project each target onto random direction.
        lengths = targets.reshape((len(targets), 2*self.n_landmarks)).dot(dir)
        cov_l_p = pixel_values.dot(lengths)/len(targets) - np.average(lengths) * pixel_averages
        correlation = (cov_l_p[:, None] - cov_l_p) / np.sqrt(np.std(lengths) * (var_pp_sum - 2 * cov_pp))

        # if np.isnan(correlation).all():
        #     return 0, 0
        res = np.nanargmax(correlation)
        return res / self.n_pixels, res % self.n_pixels

        # Random projection vector
        #projection = np.random.normal(size=136)
        # Project regression targets against random projection
        #Y_proj = targets.reshape(len(targets), 136).dot(projection)
        # Calculate covariance of projections
        #Y_proj_cov = np.cov(Y_proj, bias=1)
        # Calculate the pixel covariance
        #Y_pixels_cov  = (((pixel_values * Y_proj).sum(axis=1) / len(targets)) -
        #                 (Y_proj.sum() / len(targets)) * (pixel_values.sum(axis=1) / len(targets)))
        # Calculate the indices of the largest covariance
        #pixel_cov_diff = np.diag(cov_pp)[:, None] + np.diag(cov_pp)
        #correlation = (Y_pixels_cov[:, None] - Y_pixels_cov) / np.sqrt(
        #    Y_proj_cov * (pixel_cov_diff - 2 * cov_pp))
        #res = np.nanargmax(correlation)

        #return res/400, res%400

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
        #bins[bin_ids] += targets.reshape(len(targets), 2 * n_landmarks)
        #bins_size[bin_ids] += 1
        for i in xrange(len(bin_ids)):
            bins[bin_ids[i]] += targets[i].reshape(136,)
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
        self.compressed = False
        # Reference to the basis (if compressed).

    @staticmethod
    def get_bin(features, thresholds):
        res = 0
        for i in xrange(len(features)):
            if features[i] <= thresholds[i]:
                res |= 1 << i
        return res

    def apply(self, pixels, basis=None):
        # Select features from shape-indexed pixels.
        features = pixels[self.features[:, 0]]-pixels[self.features[:, 1]]
        # Get bin of the sample with the given shape-indexed pixels.
        bin_id = self.get_bin(features, self.thresholds)
        if not self.compressed:
            return self.bins[bin_id].reshape((self.n_landmarks, 2))
        else:
            return self._decompress_bin(bin_id, basis)

    def _decompress_bin(self, bin_id, basis):
        output = self.bins[bin_id]
        ret = np.zeros(self.n_landmarks*2)
        for (base_vector_id, coeff) in output:
            ret += basis[int(base_vector_id)]*coeff
        return ret

    def compress(self, basis, Q):
        self.compressed = True

        compressed_bins = []
        for i, current_bin in enumerate(self.bins):
            compressed_bin = np.zeros((Q, 2))
            residual = current_bin.copy()
            for i in xrange(Q):
                max_projection = 0.0
                max_i = 0

                for j, base_vector in enumerate(basis):
                    proj = residual.dot(base_vector)
                    if proj > max_projection:
                        max_projection = proj
                        max_i = j

                # max_i = np.argmax(basis.dot(residual))

                compressed_bin[i] = [max_i, 0]
                compressed_matrix = np.zeros((self.n_landmarks*2, i+1))
                for j in xrange(i+1):
                    compressed_matrix[:, j] = basis[compressed_bin[j][0]]
                compressed_matrix_t = np.transpose(compressed_matrix)


                retval, dst = cv2.solve(compressed_matrix_t.dot(compressed_matrix),
                                       compressed_matrix_t.dot(current_bin), flags=cv2.DECOMP_SVD)
                for j in xrange(i+1):
                    compressed_bin[j][1] = dst[j]
                #dst = dst.reshape(2*self.n_landmarks, 1)
                residual -= compressed_matrix.dot(dst).reshape(2*self.n_landmarks,)
            compressed_bins.append(compressed_bin)
        self.bins = compressed_bins
