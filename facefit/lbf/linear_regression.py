import site
import os
from os import path
from menpo.shape import PointCloud
from menpo.visualize import print_dynamic
import numpy as np

from facefit import util
from facefit.base import RegressorBuilder

dirname = path.dirname(path.abspath(__file__))
site.addsitedir(path.join(dirname, '../external/liblinear/python'))
import liblinearutil


class GlobalRegressionBuilder(RegressorBuilder):
    def __init__(self, feature_extractor_builder):
        self.feature_extractor_builder = feature_extractor_builder

    def build(self, images, targets, extra):
        shapes, mean_shape, i_stage = extra
        n_landmarks = mean_shape.n_points
        feature_extractor = self.feature_extractor_builder.build(images, shapes, targets, (mean_shape, i_stage))

        print("Extracting local binary features for each image.\n")
        features = [ list(feature_extractor.apply(images[i], shapes[i])) for i in xrange(len(images)) ]
        print("Features extracted.\n")
        w = np.zeros(shape=(2*n_landmarks, len(features[0])))

        for lmark in xrange(2*n_landmarks):
            print_dynamic("Learning linear regression coefficients for landmark coordinate {}/{}.\n".format(lmark, 2*n_landmarks))
            linreg = liblinearutil.train(list(targets[:, lmark]), features, "-s 12 -p 0 -c {}".format(1/float(len(features))))
            w_list = linreg.get_decfun()[0]
            w[lmark][0:len(w_list)] = w_list

        return GlobalRegression(feature_extractor, w, mean_shape)


class GlobalRegression:
    def __init__(self, feature_extractor, regression_matrix, mean_shape):
        self.feature_extractor = feature_extractor
        self.regression_matrix = regression_matrix.transpose()
        self.mean_shape = mean_shape
        self.n_landmarks = mean_shape.n_points

    def apply(self, image, shape):
        mean_to_shape = util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
        feat = self.feature_extractor.apply(image, shape)
        res = self.regression_matrix[feat == 1].sum(axis=0).reshape((self.n_landmarks, 2))
        return mean_to_shape.apply(PointCloud(res, copy=False))