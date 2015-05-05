import random
from menpo.shape import PointCloud
import numpy as np
from primitive_regressor import PrimitiveRegressor
from util import transform_to_mean_shape


class PrimaryRegressor:
    # landmarkFeatureSelector
    # primitive_regressors

    def __init__(self, P, F, nFerns, nLandmarks, mean_shape, kappa):
        self.nFerns = nFerns
        self.nFernFeatures = F
        self.nFeatures = P
        self.mean_shape = mean_shape
        self.kappa = kappa
        # TODO
        self.nPixels = P
        self.nLandmarks = nLandmarks
        # TODO: Pass argumetns for FeatureExtractorConstructor in a dictionary
        self.feature_extractor = PixelExtractor(P, nLandmarks, mean_shape, self.kappa)
        self.primitive_regressors = [PrimitiveRegressor(P, F, nFerns, nLandmarks) for i in range(nFerns)]

    def extract_features(self, img, shape, mean_to_shape, dest):
        return self.feature_extractor.extract_pixels(img, shape, mean_to_shape, dest)

    def train(self, pixel_vectors, targets):
        n_samples = len(pixel_vectors)

        # Precompute a bunch of useful numbers.
        pixel_values = np.matrix.transpose(pixel_vectors)
        cov_pp = np.cov(pixel_values)

        i = 0
        for r in self.primitive_regressors:
            print 'Training primitive regressor', i
            i += 1
            r.train(pixel_vectors, targets, cov_pp, pixel_values)
            for j in xrange(n_samples):
                targets[j] -= r.apply(pixel_vectors[j])

    def apply(self, shape, shape_indexed_features):
        res = np.zeros(shape=(self.nLandmarks, 2), dtype=float)
        for r in self.primitive_regressors:
            res += r.apply(shape_indexed_features)
        return res


class PixelExtractor:
    def __init__(self, n_pixels, n_landmarks, mean_shape, kappa):
        self.n_pixels = n_pixels
        self.pixel_coords = np.random.uniform(low=-kappa, high=kappa, size=2*n_pixels).reshape(n_pixels, 2)
        self.landmarks = np.random.randint(low=0, high=n_landmarks, size=n_pixels)
        self.mean_shape = mean_shape

    def extract_pixels(self, image, shape, mean_to_shape, dest):
        for i in xrange(self.n_pixels):
            lmark = self.landmarks[i]
            offset = mean_to_shape.apply(self.pixel_coords[i])
            x = int(shape[lmark][0] + offset[0])
            y = int(shape[lmark][1] + offset[1])

            if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
                dest[i] = 0
                continue
            # Extract pixel at (x, y).
            dest[i] = image.pixels[x][y][0]
