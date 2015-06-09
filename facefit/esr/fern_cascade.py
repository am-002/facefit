r"""
Implementation of a cascade of ferns.
"""

from menpo.shape import PointCloud
import numpy as np
from menpo.visualize import print_dynamic
from facefit.esr.fern import FernBuilder

from facefit.inner_cascade import InnerCascadeBuilder
from facefit import util


class FernCascadeBuilder(InnerCascadeBuilder):
    def __init__(self, feature_extractor_builder, fern_builder, n_ferns,
                  compress, basis_size, compression_maxnonzero):
        super(self.__class__, self).__init__(n_primitive_regressors=n_ferns, primitive_builder=fern_builder,
                                             feature_extractor_builder=feature_extractor_builder)
        self.basis_size = basis_size
        self.compression_maxnonzero = compression_maxnonzero
        self.compress = compress

    def _random_basis(self, ferns, basis_size):
        n_ferns = len(ferns)
        n_fern_features = len(ferns[0].features)
        output_indices = np.random.choice(a=n_ferns*(1 << n_fern_features),
                                          size=basis_size, replace=False)
        basis = []
        for i, j in enumerate(output_indices):
            fern_id = j / (1 << n_fern_features)
            bin_id = j % (1 << n_fern_features)
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

