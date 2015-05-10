import numpy as np

BETA = 0

class Fern:
    r"""
    Implementation of a random Fern.

    """
    def __init__(self, n_pixels, n_fern_features, n_ferns, n_landmarks):
        self.n_fern_features = n_fern_features
        self.fern_feature_selector = FeatureSelector(n_pixels, n_fern_features)
        self.bins = np.zeros((2**n_fern_features, 2*n_landmarks))
        self.bins_size = np.zeros(2**n_fern_features)

    def get_bin(self, feature_vector):
        res = 0
        for i in range(self.n_fern_features):
            if feature_vector[i] <= self.thresholds[i]:
                res |= 1 << i
        return res

    def train(self, feature_vectors, targets, cov_pp, pixels, pixel_averages, pixel_var_sum):
        self.fern_feature_selector.train(targets, cov_pp, pixels, pixel_averages, pixel_var_sum)

        fern_feature_vectors = np.apply_along_axis(self.fern_feature_selector.extract_features, axis=1, arr=feature_vectors)

        ranges_min = fern_feature_vectors.min(axis=0)
        ranges_max = fern_feature_vectors.max(axis=0)

        self.thresholds = np.random.uniform(low=ranges_min, high=ranges_max)

        bins = np.apply_along_axis(self.get_bin, arr=fern_feature_vectors, axis=1)
        self.bins[bins] += targets
        self.bins_size[bins] += 1

    def apply(self, shape_indexed_features):
        fern_features = self.fern_feature_selector.extract_features(shape_indexed_features)
        bin_id = self.get_bin(fern_features)
        if self.bins_size[bin_id] == 0.0:
            return self.bins[bin_id]
        return self.bins[bin_id] / (self.bins_size[bin_id] + BETA)

class FeatureSelector:
    def __init__(self, n_pixels, n_fern_features):
        self.n_pixels=n_pixels
        self.n_fern_features = n_fern_features
        self.features = np.zeros((2, n_fern_features), order='C', dtype=int)

    def train(self, targets, cov_pp, pixels, pixel_averages, pixel_var_sum):
        n_images = len(targets)
        n_landmarks = len(targets[0])/2
        # n_pixels = len(pixels[0])
        n_pixels = self.n_pixels

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

            correlation = (cov_l_p[:, None] - cov_l_p) / np.sqrt(
                lengths_std * (pixel_var_sum - 2 * cov_pp))
            if np.isnan(correlation).all():
                # If we reach this point, all residual targets are probably zero, thus we just quit.
                return
            res = np.nanargmax(correlation)

            self.features[0][f] = res/n_pixels
            self.features[1][f] = res%n_pixels

            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).
            # Here the features array is populated

    def extract_features(self, pixels):
        return pixels[self.features[0]]-pixels[self.features[1]]
