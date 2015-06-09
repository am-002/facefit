from facefit import cascade
from facefit.pixel_extractor import PixelExtractorBuilder
from fern import FernBuilder
from fern_cascade import FernCascadeBuilder


class ESRBuilder(cascade.CascadedShapeRegressorBuilder):
    def __init__(self, n_landmarks=68, n_stages=10,  n_perturbations=20, n_ferns=500, n_pixels=400, kappa=0.3,
                 n_fern_features=5, beta=1000, compress_ferns=True, basis_size=512, compression_maxnonzero=5):

        feature_extractor_builder = PixelExtractorBuilder(n_landmarks=n_landmarks, n_pixels=n_pixels, kappa=kappa)
        fern_builder = FernBuilder(n_features=n_fern_features, beta=beta)
        cascade_builder = FernCascadeBuilder(feature_extractor_builder=feature_extractor_builder,
                                             fern_builder=fern_builder, n_ferns=n_ferns,
                                             compress=compress_ferns, basis_size=basis_size,
                                             compression_maxnonzero=compression_maxnonzero)

        super(self.__class__, self).__init__(n_stages=n_stages, n_perturbations=n_perturbations,
                                             weak_builder=cascade_builder)