import sys

from menpo.shape import PointCloud
import menpo.io as mio

from primary_regressor import PrimaryRegressor
import util
import numpy as np
from util import *

class ExplicitShapeRegressor:
    def __init__(self, n_landmarks, n_regressors=10, n_pixels=400, n_fern_features=5, n_ferns=500, n_perturbations=20):
        self.n_landmarks = n_landmarks
        self.n_regressors = n_regressors
        self.n_pixels = n_pixels
        self.n_ferns = n_ferns
        self.n_perturbations = n_perturbations

        # Calculate mean shape from a subset of training data.
        self.mean_shape = getNormalisedMeanShape('../helen/subset_cropped/')

        # TODO: In the paper, the kappa was set to 0.3*distance-between-eye-pupils-in-mean-shape.
        self.kappa = 0.3

        self.regressors = [PrimaryRegressor(n_pixels, n_fern_features, n_ferns, n_landmarks, self.mean_shape, self.kappa)
                                for i in range(n_regressors)]

    def train(self, img_glob):

        images = []
        for img in mio.import_images(img_glob):
            if img.has_landmarks:
                images.append(img.as_greyscale())

        print 'Finished reading input'

        n_samples = len(images)
        shapes = np.ndarray(shape=(n_samples*self.n_perturbations, self.n_landmarks, 2), dtype=float)

        pixels = np.ndarray(shape=(n_samples*self.n_perturbations, self.n_pixels), dtype=float)
        targets = np.ndarray(shape=(n_samples*self.n_perturbations, self.n_landmarks, 2), dtype=float)
        shape_to_mean = [None for i in xrange(n_samples*self.n_perturbations)]
        mean_to_shape = [None for i in xrange(n_samples*self.n_perturbations)]

        print 'Finished allocating memory'

        for i, img in enumerate(images):
            bounding_box = get_bounding_box(img)
            index = i*self.n_perturbations
            shapes[index] = fit_shape_to_box(self.mean_shape, bounding_box).points
            for j in xrange(index, index+self.n_perturbations):
                shapes[j] = fit_shape_to_box(self.mean_shape, perturb(bounding_box)).points

        # TODO: Read everything into memory.
        # 1. scale down OR
        # 2. crop to face
        # 3. calc mean out of trainset
        # 4. maybe convert from 64bit to 32bit float?
        for regressor_i, regressor in enumerate(self.regressors):
            print 'Training primary regressor ', regressor_i
            for i, img in enumerate(images):
                for j in xrange(self.n_perturbations):
                    index = i*self.n_perturbations + j

                    delta = PointCloud(img.landmarks['PTS'].lms.points - shapes[index], copy=False)
                    shape_to_mean[index] = util.transform_to_mean_shape(shapes[index], self.mean_shape)
                    mean_to_shape[index] = shape_to_mean[index].pseudoinverse()

                    normalized_target = shape_to_mean[index].apply(delta).points

                    targets[index] = normalized_target
                    regressor.extract_features(img, shapes[index], mean_to_shape[index], pixels[index])

            regressor.train(pixels, targets)

            for i in xrange(len(shapes)):
                normalized_offset = regressor.apply(shapes[i], pixels[i])
                offset = mean_to_shape[i].apply(normalized_offset)
                shapes[i] += offset

    def fit(self, image, initial_shape):
        image = image.as_greyscale()
        shape = fit_shape_to_box(initial_shape, get_bounding_box(image))

        pixels = np.zeros(shape=(self.n_pixels))
        for r in self.regressors:
            shape_to_mean = util.transform_to_mean_shape(shape.points, self.mean_shape)
            r.extract_features(image, shape.points, shape_to_mean, pixels)
            offset = shape_to_mean.pseudoinverse().apply(r.apply(shape.points, pixels))
            shape.points += offset
            #shape.points += r.apply(shape, pixels).points

        return shape
