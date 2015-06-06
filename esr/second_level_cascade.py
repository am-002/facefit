from abc import abstractmethod
from menpo.visualize import print_dynamic
import util
from menpo.shape import PointCloud

__author__ = 'andrejm'

import numpy as np

class SecondLevelCascadeBuilder(object):
    def __init__(self, n_primitive_regressors, n_landmarks, primitive_builder, feature_extractor_builder):
        self.n_primitive_regressors = n_primitive_regressors
        self.n_landmarks = n_landmarks
        self.primitive_builder = primitive_builder
        self.feature_extractor_builder = feature_extractor_builder

    @abstractmethod
    def precompute(self, pixel_vectors, feature_extractor, mean_shape):
        return None

    @abstractmethod
    def post_process(self, primitive_regressors):
        return None

    #TODO: get rid of this function
    def to_mean(self, shape):
        return util.transform_to_mean_shape(shape, self.mean_shape)

    def build(self, images, shapes, gt_shapes, mean_shape, i_stage):
        self.mean_shape = mean_shape
        assert(len(images) == len(shapes))
        assert(len(shapes) == len(gt_shapes))

        # Calculate normalized targets.
        deltas = [gt_shape.points - shape.points for gt_shape, shape in zip(gt_shapes, shapes)]
        targets = np.array([self.to_mean(shape).apply(delta).reshape((2*self.n_landmarks,)) for (shape, delta) in zip(shapes, deltas)])

        feature_extractor = self.feature_extractor_builder.build(images, targets, i_stage)

        # Extract shape-indexed pixels from images.
        pixel_vectors = np.array([feature_extractor.extract_features(img, shape, self.to_mean(shape).pseudoinverse())
                                  for (img, shape) in zip(images, shapes)])

        data = self.precompute(pixel_vectors, feature_extractor, mean_shape)

        primitive_regressors = []
        for i in xrange(self.n_primitive_regressors):
            print_dynamic("Building primitive regressor {}".format(i))
            primitive_regressor = self.primitive_builder.build(pixel_vectors, targets, data)
            # Update targets.
            targets -= [primitive_regressor.apply(pixel_vector, data) for pixel_vector in pixel_vectors]
            primitive_regressors.append(primitive_regressor)

        post_data = self.post_process(primitive_regressors)

        return SecondLevelCascade((self.n_landmarks, feature_extractor, primitive_regressors, mean_shape, post_data))

class SecondLevelCascade:
    def __init__(self, data):
        n_landmarks, feature_extractor, ferns, mean_shape, extra = data
        self.n_landmarks = n_landmarks
        self.feature_extractor = feature_extractor
        self.ferns = ferns
        self.mean_shape = mean_shape
        self.extra = extra

    def apply(self, image, shape):
        mean_to_shape = util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
        shape_indexed_features = self.feature_extractor.extract_features(image, shape, mean_to_shape)
        res = PointCloud(np.zeros((self.n_landmarks, 2)), copy=False)
        for r in self.ferns:
            offset = r.apply(shape_indexed_features, self.extra)
            res.points += offset.reshape((self.n_landmarks, 2))
        return mean_to_shape.apply(res)
