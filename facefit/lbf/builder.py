from facefit.lbf.feature_extractor import LocalBinaryFeaturesExtractorBuilder
from facefit.lbf.linear_regression import GlobalRegressionBuilder
import numpy as np
from facefit import cascade


class LBFBuilder(cascade.CascadedShapeRegressorBuilder):
    # More accurate (but slower version): n_stages = 5, n_trees = 1200, depth = 7;
    # faster version: (n_stages = 5, n_trees = 300, depth = 5)

    def __init__(self, n_landmarks=68, n_stages=5, n_trees=300, tree_depth=5,
                 n_perturbations=20, n_pixels=400, n_tree_test_features=500, kappa=0.3, MU=0.5):
        n_trees_per_landmark = int(np.ceil(n_trees / float(n_landmarks)))

        lbf_extractor_builder = LocalBinaryFeaturesExtractorBuilder(n_pixels=n_pixels,
                                                                    kappa=kappa,
                                                                    n_trees_per_landmark=n_trees_per_landmark,
                                                                    tree_depth=tree_depth,
                                                                    n_tree_test_features=n_tree_test_features,
                                                                    MU=MU)

        lbf_regressor_builder = GlobalRegressionBuilder(lbf_extractor_builder)
        super(self.__class__, self).__init__(n_stages=n_stages, n_perturbations=n_perturbations,
                                             weak_builder=lbf_regressor_builder)
