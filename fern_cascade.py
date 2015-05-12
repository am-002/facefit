from menpo.shape import PointCloud
import numpy as np
from fern import FernBuilder
import util
from menpo.visualize import print_dynamic

class FernCascadeBuilder:
    def __init__(self, n_pixels, n_fern_features, n_ferns, n_landmarks, mean_shape, kappa, beta):
        self.n_ferns = n_ferns
        self.n_fern_features = n_fern_features
        self.n_features = n_pixels
        self.kappa = kappa
        self.n_pixels = n_pixels
        self.n_landmarks = n_landmarks
        self.beta = beta
        self.mean_shape = mean_shape

    def to_mean(self, shape):
        return util.transform_to_mean_shape(shape, self.mean_shape)

    def build(self, images, shapes, gt_shapes):
        assert(len(images) == len(shapes))
        assert(len(shapes) == len(gt_shapes))

        feature_extractor = FeatureExtractor(self.n_landmarks, self.n_pixels, self.kappa)

        # Calculate normalized targets.
        deltas = [gt_shape.points - shape.points for gt_shape, shape in zip(gt_shapes, shapes)]

        # print 'True delta is ', deltas[0]
        # print 'True shape is ', images[0].landmarks['PTS'].lms.points

        targets = np.array([self.to_mean(shape).apply(delta) for (shape, delta) in zip(shapes, deltas)])
        # print 'True normalized delta is ', targets[0]

        # print 'deltas[0] ', deltas[0]
        # print 'deltas[1] ', deltas[1]
        # print 'targets[0] ', targets[0]
        # print 'targets[1] ', targets[1]

        # Extract shape-indexed pixels from images.
        pixel_vectors = np.array([feature_extractor.extract_features(img, shape, self.to_mean(shape).pseudoinverse())
                                  for (img, shape) in zip(images, shapes)])

        # print 'Extracted features in builder: ', pixel_vectors[0]

        # Precompute values common for all ferns.
        pixel_vals = np.transpose(pixel_vectors)
        pixel_averages = np.average(pixel_vals, axis=1)
        cov_pp = np.cov(pixel_vals, bias=1)
        pixel_var_sum = np.diag(cov_pp)[:, None] + np.diag(cov_pp)

        ferns = []
        for i in xrange(self.n_ferns):
            print_dynamic("Building fern {}".format(i))
            fern_builder = FernBuilder(self.n_pixels, self.n_fern_features, self.n_landmarks, self.beta)
            fern = fern_builder.build(pixel_vectors, targets, cov_pp, pixel_vals, pixel_averages, pixel_var_sum)
            # Update targets.
            targets -= [fern.apply(pixel_vector) for pixel_vector in pixel_vectors]
            ferns.append(fern)
        return FernCascade(self.n_landmarks, feature_extractor, ferns)


class FeatureExtractor:
    def __init__(self, n_landmarks, n_pixels, kappa):
        self.lmark = np.random.randint(low=0, high=n_landmarks, size=n_pixels)
        self.pixel_coords = np.random.uniform(low=-kappa, high=kappa, size=n_pixels*2).reshape(n_pixels, 2)

    def extract_features(self, img, shape, mean_to_shape):
        offsets = mean_to_shape.apply(self.pixel_coords)
        ret = shape.points[self.lmark] + offsets
        # if "10405146_1" in str(img.path):
        #     print 'Indexing with shape ', shape.points
        #     print 'mean_to_shape ', mean_to_shape
        #     print 'got offsets ',offsets
        return util.sample_image(img, ret)


class FernCascade:
    def __init__(self, n_landmarks, feature_extractor, ferns):
        self.n_landmarks = n_landmarks
        self.feature_extractor = feature_extractor
        self.ferns = ferns

    def apply(self, image, shape, mean_to_shape):
        shape_indexed_features = self.feature_extractor.extract_features(image, shape, mean_to_shape)
        res = PointCloud(np.zeros((self.n_landmarks, 2)), copy=False)
        for r in self.ferns:
            offset = r.apply(shape_indexed_features)
            res.points += offset.reshape((68, 2))
        # if "10405146_1" in str(image.path):
        #     print 'Extracted features in apply: ', shape_indexed_features
        #     print 'Got normalized offset ', res.points
        return mean_to_shape.apply(res)
