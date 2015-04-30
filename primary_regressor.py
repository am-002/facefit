import random
from menpo.shape import PointCloud
from menpo.transform import AlignmentSimilarity
import numpy as np
from primitive_regressor import PrimitiveRegressor
from util import transform_to_mean_shape


class PrimaryRegressor:
    # landmarkFeatureSelector
    # primitive_regressors

    def __init__(self, P, F, nFerns, nLandmarks, mean_shape):
        self.nFerns = nFerns
        self.nFernFeatures = F
        self.nFeatures = P
        self.mean_shape = mean_shape
        # TODO
        self.nPixels = P
        self.nLandmarks = nLandmarks
        # TODO: Pass argumetns for FeatureExtractorConstructor in a dictionary
        self.feature_extractor = PixelExtractor(P, nLandmarks, mean_shape)
        self.primitive_regressors = [PrimitiveRegressor(P, F, nFerns, nLandmarks) for i in range(nFerns)]

    def extract_features(self, img, shape):
        return self.feature_extractor.extract_pixels(img, shape, transform_to_mean_shape(shape, self.mean_shape).pseudoinverse())

    def train(self, pixel_vectors, targets):
        n_samples = len(pixel_vectors)
        ##print pixel_vectors
        pixel = zip(*pixel_vectors)

        #print 'pixels: ', pixel

        pixels_sum = map(sum, pixel)
        cov_pp = np.cov(pixel)

        i = 0
        for r in self.primitive_regressors:
            #print 'Training regressor ', i
            i += 1
            r.train(pixel_vectors, targets, cov_pp, pixels_sum, pixel)
            for j in xrange(n_samples):
                targets[j].points -= r.apply(pixel_vectors[j]).points
                #print targets[j].points

    def apply(self, shape, shape_indexed_features):
        res = PointCloud([[0.0, 0.0] for i in range(self.nLandmarks)])
        #mean_to_shape = AlignmentSimilarity(self.mean_shape, shape)

        for r in self.primitive_regressors:
            #offset = mean_to_shape.apply(r.apply(shape_indexed_features))
            offset = r.apply(shape_indexed_features)
            res.points += offset.points
        return res


class PixelExtractor:
    def __init__(self, n_pixels, n_landmarks, mean_shape):
        self.n_pixels = n_pixels
        self.pixel_coords = []
        self.mean_shape = mean_shape

        for i in xrange(n_pixels):
            lmark = random.randint(0, n_landmarks-1)
            #TODO: normalize this.
            dx = random.uniform(-30, 30)
            dy = random.uniform(-30, 30)
            self.pixel_coords.append((lmark, [dx, dy]))

    def extract_pixels(self, image, shape, mean_to_shape):
        ret = []
        for (lmark, offset) in self.pixel_coords:
            # TODO: store normalized offsets
            #offset = mean_to_shape.apply(offset)
            x = int(shape.points[lmark][0] + offset[0])
            y = int(shape.points[lmark][1] + offset[1])

            if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
                ret.append(0)
                continue

            # TODO: Assuming its greyscale.
            ret.append(image.pixels[x][y][0])

        return ret