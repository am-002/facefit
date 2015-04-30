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

    def train(self, feature_vectors, targets, cov_pp, pixels_sum, pixels):
        self.fern_feature_selector.train(feature_vectors, targets, cov_pp, pixels_sum, pixels)
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
            self.bins[bin_id].add(targets[i])

    def test(self, feature_vector):
        bin_id = self.get_bin(feature_vector)
        return self.bins[bin_id].get_delta()


class FeatureSelector:
    def __init__(self, nPixels, F):
        self.nPixels = nPixels
        self.F = F
        self.features = [(0,0)]*F
        random.seed(None)

    def average(self, x):
        return float(sum(x)) / len(x)

    # TODO: Random projection from unit gaussian?
    def getRandomDirection(self):
        return np.array([random.uniform(-1, 1) for i in range(2*68)])

    def magnitude(self, x):
        return np.sqrt(np.dot(x, x))

    # Project a onto b and return length of the projection.
    def project(self, a, b):
        return np.dot(a, b) / (self.magnitude(a) * self.magnitude(b));

    def cov(self, x, y):
        mean_x = self.average(x)
        mean_y = self.average(y)
        res = 0.0
        for i in range(len(x)):
            res += (x[i]-mean_x) * (y[i]-mean_y)
        return res / float(len(x))


    def train(self, features, targets, cov_pp, pixel_sum, pixels):


        for f in xrange(self.F):
            #print 'COV_PP: ', cov_pp
            # Project the offset in random direction and measure the scalar length
            # of the projection. Find which feature has highest imapct on the length.

            d = self.getRandomDirection()

            l = []

            for target in targets:
                l.append(self.project(target.as_vector(), d))
            #print 'l is ', l

            l_sum = np.sum(l)
            var_l = self.cov(l, l)

            # set l to l[i]-l_mean as a speed up!

            cov_l_p = [0]*self.nPixels
            n_samples = float(len(features))
            n_pixels = self.nPixels
            # for i in xrange(len(features)):
            #     li = l[i]
            #     for pixel in xrange(n_pixels):
            #         cov_l_p[pixel] = features[i][pixel]*li
            #
            # for pixel in xrange(n_pixels):
            #     cov_l_p[pixel] = (cov_l_p[pixel] + pixel_sum[pixel]*l_sum*(n_samples-2/n_samples))/n_samples
            #
            # for pixel in xrange(n_pixels):
            #     cov_l_p[pixel] /= float(n_pixels-1)

            for i in xrange(self.nPixels):
                cov_l_p[i] = self.cov(l, pixels[i])

            #print 'var(l) = ', var_l
            #print 'cov_l_p = ', cov_l_p

            #    cov_l_p[pixel] = self.cov(l, p)


            var_p = [cov_pp[i][i] for i in xrange(self.nPixels)]

            maxcorr = -12345
            res = (0, 0)
            for i in xrange(n_pixels):
                cov_ppii = var_p[i]
                cov_l_pi = cov_l_p[i]
                for j in xrange(i+1, n_pixels):
                    # We want to get corr(l, pixel_i - pixel_j).
                    denom = var_l*(cov_ppii + var_p[j]-2*cov_pp[i][j])
                    if (denom <= 0):
                        # TODO: What to do in this case?
                        continue
                    # TODO: abs here?
                    corr = (cov_l_pi - cov_l_p[j]) / np.sqrt(denom)
                    #print corr
                    if (corr < 0):
                        corr = -corr

                    if corr > maxcorr:
                        maxcorr = corr
                        res = (i, j)
            self.features[f] = res

            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).
            # Here the features array is populated

    def extract_features(self, pixels):
        return [pixels[p[0]] - pixels[p[1]] for p in self.features]
