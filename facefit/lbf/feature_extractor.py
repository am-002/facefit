from menpo.visualize import print_dynamic
import numpy as np
from facefit.base import FeatureExtractor, FeatureExtractorBuilder

from facefit.ert import forest
from facefit import util
from facefit.ert.tree import RegressionTreeBuilder
from facefit.pixel_extractor import PixelExtractorBuilder

class LocalBinaryFeaturesExtractor(FeatureExtractor):
    def __init__(self, forests, mean_shape):
        self.forests = forests
        self.mean_shape = mean_shape

    def apply(self, img, shape):
        n_landmarks = len(self.forests)
        n_trees = len(self.forests[0].regressors)
        n_leaves = len(self.forests[0].regressors[0].leaves)
        local_binary_features = np.zeros(n_landmarks*n_trees*n_leaves)
        mean_to_shape = util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()

        for landmark_i, f in enumerate(self.forests):
            pixels = f.feature_extractor.extract_features(img, shape, mean_to_shape)
            for tree_i, tree in enumerate(f.regressors):
                leaf = tree.get_leaf_index(pixels)
                local_binary_features[landmark_i*n_trees*n_leaves + tree_i*n_leaves + leaf] = 1
        return local_binary_features

    def get_indices(self, img, shape, mean_to_shape):
        k = 0
        n_trees = len(self.forests[0].regressors)
        n_leaves = len(self.forests[0].regressors[0].leaves)
        base_ptr = 0
        bf_per_landmark = n_trees*n_leaves

        n_landmarks = self.mean_shape.n_points
        ret = np.zeros(n_landmarks*n_trees, dtype=int)

        for f in self.forests:
            pixels = f.feature_extractor.extract_features(img, shape, mean_to_shape)
            for tree_i, tree in enumerate(f.regressors):
                ret[k] = base_ptr + tree_i*n_leaves + tree.get_leaf_index(pixels)
                k += 1
            base_ptr += bf_per_landmark
        return ret




class LocalBinaryFeaturesExtractorBuilder(FeatureExtractorBuilder):
    def __init__(self, n_pixels, kappa, n_trees_per_landmark, tree_depth, n_tree_test_features, MU):
        self.n_pixels = n_pixels
        self.kappa = kappa
        self.n_trees_per_landmark = n_trees_per_landmark
        self.tree_depth = tree_depth
        self.n_tree_test_features = n_tree_test_features
        self.MU = MU

    def build(self, images, shapes, targets, extra):
        mean_shape, i_stage = extra
        n_landmarks = mean_shape.n_points
        targets = targets.copy()

        MU = self.MU

        forests = []
        for i in xrange(n_landmarks):
            print_dynamic('Building forest for landmark {}.\n'.format(i))
            landmark_targets = targets[:, 2*i:(2*i+1)+1]
            feature_extractor_builder = PixelExtractorBuilder(n_landmarks=1, n_pixels=self.n_pixels,
                                                              kappa=self.kappa, adaptive=True,  around_landmark=i)

            tree_builder = RegressionTreeBuilder(MU=MU, depth=self.tree_depth, n_test_features=self.n_tree_test_features,
                                                 exponential_prior=False)

            forest_builder = forest.RegressionForestBuilder(n_trees=self.n_trees_per_landmark, tree_builder=tree_builder,
                                                            feature_extractor_builder=feature_extractor_builder)

            f = forest_builder.build(images, landmark_targets, (shapes, mean_shape, i_stage))
            forests.append(f)

        return LocalBinaryFeaturesExtractor(forests, mean_shape)
