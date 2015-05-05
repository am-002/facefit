import numpy as np
import weave

INF = 102345678
# BETA = 1000
BETA = 0

class PrimitiveRegressor:
    def __init__(self, n_features, n_fern_features, n_ferns, n_landmarks):
        self.nFerns = n_ferns
        self.nFernFeatures = n_fern_features
        self.nFeatures = n_features
        self.nLandmarks = n_landmarks
        self.fern = Fern(n_fern_features, n_landmarks)
        self.fern_feature_selector = FeatureSelector(n_features, n_fern_features)

    def train(self, feature_vectors, targets, cov_pp, pixels):
        self.fern_feature_selector.train(feature_vectors, targets, cov_pp, pixels)

        fern_feature_vectors = np.ndarray(shape=(len(feature_vectors), self.nFernFeatures), dtype=float)
        for i in xrange(len(feature_vectors)):
             self.fern_feature_selector.extract_features(feature_vectors[i], fern_feature_vectors[i])

        self.fern.train(fern_feature_vectors, targets)

    def apply(self, shape_indexed_features):
        fern_features = np.zeros(shape=(self.nFernFeatures,))
        self.fern_feature_selector.extract_features(shape_indexed_features, fern_features)

        return self.fern.apply(fern_features)


class Fern:
    r"""
    Implementation of a random Fern.

    """

    def __init__(self, n_features, n_landmarks):
        self.n_features = n_features
        self.thresholds = np.zeros(n_features, dtype=float)
        self.bins = np.zeros(shape=(2**n_features, 68, 2), dtype=float)
        self.bin_sz = np.zeros(shape=(2**n_features), dtype=float)

    def get_bin(self, feature_vector):
        thresholds = self.thresholds
        n_features = self.n_features
        code = """
            int res = 0;
            for (int i = 0; i < n_features; ++i) {
                if (feature_vector[i] <= thresholds[i])
                    res |= 1 << i;
            }
            return_val = res;
        """
        return weave.inline(code, ['thresholds', 'n_features', 'feature_vector'])

    def train(self, feature_vectors, targets):
        n_samples = len(feature_vectors)

        # Generate thresholds for each feature. We first get the ranges
        # of all features in the training set.
        ranges = [(INF, -INF)] * self.n_features
        #ranges = np.ndarray(shape=(self.n_features, 2))

        for feature_vector in feature_vectors:
           for i, feature in enumerate(feature_vector):
               ranges[i] = (min(ranges[i][0], feature), max(ranges[i][1], feature))

        for i, rng in enumerate(ranges):
            self.thresholds[i] = np.random.uniform(low=rng[0], high=rng[1])

        for i in xrange(n_samples):
            bin_id = self.get_bin(feature_vectors[i])
            self.bins[bin_id] += targets[i]
            self.bin_sz[bin_id] += 1

    def apply(self, feature_vector):
        bin_id = self.get_bin(feature_vector)

        if self.bin_sz[bin_id] == 0.0:
            return np.zeros(shape=(68, 2), dtype=float)

        return self.bins[bin_id] / (self.bin_sz[bin_id] + BETA)


class FeatureSelector:
    def __init__(self, n_pixels, n_fern_features):
        self.nPixels = n_pixels
        self.n_fern_features = n_fern_features
        self.features = np.zeros(shape=(n_fern_features, 2))

    def train(self, pixel_vectors, targets, cov_pp, pixel_values):
        l = np.zeros(len(targets), dtype=float)

        for f in xrange(self.n_fern_features):
            # Project the offset in random direction and measure the scalar length
            # of the projection. Find which feature has highest impact on the length.

            # Get random direction.
            dir = np.random.randn(136)

            # Normalize it.
            dir /= np.linalg.norm(dir)

            # Calculate projections of each target onto the random direction.
            l = targets.reshape((len(targets), 136)).dot(dir)

            # Calculate variance of projections.
            var_l = np.var(l)
            n_pixels = self.nPixels

            # Calculate covariances between projections and each pixel.
            cov_l_p = np.zeros(shape=(self.nPixels), dtype=float)
            for i in xrange(self.nPixels):
                cov_l_p[i] = np.cov(m=l, y=pixel_values[i])[0][1]

            max_corr_code = """
                float maxcorr = -12345;
                int res = -1;
                double varl = var_l;

                for (int i = 0; i < n_pixels-1; ++i) {
                    double var_pii = cov_pp[i*n_pixels+i];
                    int row = i*n_pixels;
                    for (int j = i+1; j < n_pixels; ++j) {
                        double denom = varl*(var_pii + cov_pp[j*n_pixels]-2*cov_pp[row+j]);
                        if (denom <= 0) {
                            continue;
                        }
                        float corr = (cov_l_p[i] - cov_l_p[j]) / sqrt(denom);
                        if (corr < 0) {
                            corr = -corr;
                        }
                        if (corr > maxcorr) {
                            maxcorr = corr;
                            res = i*n_pixels + j;
                        }
                    }
                }
                return_val = res;
            """

            res = weave.inline(max_corr_code, ['cov_pp', 'n_pixels', 'cov_l_p', 'var_l'])

            # Extract pixel pair from res and add it to self.features.
            self.features[f] = (res/n_pixels, res%n_pixels)

            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).

    def extract_features(self, pixels, dest):
        for i in xrange(self.n_fern_features):
            dest[i] = pixels[self.features[i][0]] - pixels[self.features[i][1]]
        #return [pixels[p[0]] - pixels[p[1]] for p in self.features]
