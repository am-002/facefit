from facefit import cascade
from facefit.pixel_extractor import PixelExtractorBuilder
from tree import RegressionTreeBuilder
from forest import RegressionForestBuilder

class ERTBuilder(cascade.CascadedShapeRegressorBuilder):
    def __init__(self, n_landmarks=68, n_stages=10, n_trees=500, tree_depth=5, n_candidate_splits=20,
                 exponential_prior=True, n_perturbations=20, n_pixels=400, kappa=0.3, MU=0.1):

        feature_extractor_builder = PixelExtractorBuilder(n_landmarks=n_landmarks, n_pixels=n_pixels, kappa=kappa)

        tree_builder = RegressionTreeBuilder(depth=tree_depth, n_test_features=n_candidate_splits,
                                             exponential_prior=exponential_prior, MU=MU)

        forest_builder = RegressionForestBuilder(n_trees=n_trees, tree_builder=tree_builder,
                                                 feature_extractor_builder=feature_extractor_builder)

        super(self.__class__, self).__init__(n_stages=n_stages, n_perturbations=n_perturbations,
                                             weak_builder=forest_builder)
