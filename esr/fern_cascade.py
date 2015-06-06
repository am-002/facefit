r"""
Implementation of a cascade of ferns.
"""

from menpo.shape import PointCloud
import numpy as np
from menpo.visualize import print_dynamic
from esr.second_level_cascade import SecondLevelCascade, SecondLevelCascadeBuilder

from fern import FernBuilder
import util
from util import PixelExtractor

class FernCascadeBuilder(SecondLevelCascadeBuilder):
    def __init__(self, primitive_builder, feature_extractor_builder, n_landmarks=68, n_pixels=400, n_ferns=500, n_fern_features=5, kappa=0.3, beta=1000,
                 compress=True, basis_size=512, compression_maxnonzero=5):
        super(self.__class__, self).__init__(n_ferns, n_landmarks, primitive_builder, feature_extractor_builder)
        self.n_ferns = n_ferns
        self.n_fern_features = n_fern_features
        self.n_features = n_pixels
        self.kappa = kappa
        self.n_pixels = n_pixels
        self.n_landmarks = n_landmarks
        self.beta = beta
        self.mean_shape = None
        self.basis_size = basis_size
        self.compression_maxnonzero = compression_maxnonzero
        self.compress = compress

    def _random_basis(self, ferns, basis_size):
        output_indices = np.random.choice(a=self.n_ferns*(1 << self.n_fern_features), size=basis_size, replace=False)

        basis = []
        for i, j in enumerate(output_indices):
            fern_id = j / (1 << self.n_fern_features)
            bin_id = j % (1 << self.n_fern_features)
            ret = ferns[fern_id].bins[bin_id].copy()
            ret = util.normalize(ret)
            basis.append(ret)
        return np.array(basis)

    def precompute(self, pixel_vectors, feature_extractor, mean_shape):
        # Precompute values common for all ferns.
        pixel_vals = np.transpose(pixel_vectors)
        pixel_averages = np.average(pixel_vals, axis=1)
        cov_pp = np.cov(pixel_vals, bias=1)
        pixel_var_sum = np.diag(cov_pp)[:, None] + np.diag(cov_pp)
        return cov_pp, pixel_vals, pixel_averages, pixel_var_sum

    def post_process(self, ferns):
        basis = None
        if self.compress:
            print("\nPerforming fern compression.\n")
            # Create a new basis by randomly sampling from all fern outputs.
            basis = self._random_basis(ferns, self.basis_size)
            for i, fern in enumerate(ferns):
                print_dynamic("Compressing fern {}/{}.".format(i, len(ferns)))
                fern.compress(basis, self.compression_maxnonzero)
        return basis

class FernCascade:
    def __init__(self, n_landmarks, feature_extractor, ferns, basis, mean_shape):
        self.n_landmarks = n_landmarks
        self.feature_extractor = feature_extractor
        self.ferns = ferns
        self.basis = basis
        self.mean_shape = mean_shape

    def apply(self, image, shape):
        mean_to_shape = util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
        shape_indexed_features = self.feature_extractor.extract_features(image, shape, mean_to_shape)
        res = PointCloud(np.zeros((self.n_landmarks, 2)), copy=False)
        for r in self.ferns:
            offset = r.apply(shape_indexed_features, self.basis)
            res.points += offset.reshape((self.n_landmarks, 2))
        return mean_to_shape.apply(res)
