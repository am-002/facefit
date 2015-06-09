from facefit.inner_cascade import InnerCascadeBuilder


class RegressionForestBuilder(InnerCascadeBuilder):
    def __init__(self, n_trees, tree_builder, feature_extractor_builder):
        super(self.__class__, self).__init__(n_trees, tree_builder, feature_extractor_builder)

    def precompute(self, pixel_vectors, pixel_extractor, mean_shape):
        pixel_coords = mean_shape.points[pixel_extractor.lmark] + pixel_extractor.pixel_coords
        return pixel_coords, mean_shape


