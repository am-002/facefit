import sys

from menpo.shape import PointCloud
import menpo.io as mio

from primary_regressor import PrimaryRegressor
import util
from util import *
import copy


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
        shapes = []

        # TODO: Read everything into memory.
        # 1. scale down OR
        # 2. crop to face
        # 3. calc mean out of trainset
        # 4. maybe convert from 64bit to 32bit float?
        for regressor_i, regressor in enumerate(self.regressors):
            pixels = []
            targets = []

            img_i = 0
            for img_orig in mio.import_images(img_glob):
                if not img_orig.has_landmarks:
                    continue
                # Convert to greyscale
                img = img_orig.as_greyscale()

                if regressor_i == 0:
                    bounding_box = get_bounding_box(img)
                    shapes.append(fit_shape_to_box(self.mean_shape, bounding_box))
                    for j in xrange(self.n_perturbations-1):
                        shapes.append(fit_shape_to_box(self.mean_shape, perturb(bounding_box)))

                for j in xrange(self.n_perturbations):
                    index = img_i*self.n_perturbations + j

                    delta = PointCloud(img.landmarks['PTS'].lms.points - shapes[index].points)
                    normalized_target = util.transform_to_mean_shape(shapes[index], self.mean_shape).apply(delta)

                    targets.append(normalized_target)
                    pixels.append(regressor.extract_features(img, shapes[index]))

                img_i += 1


            regressor.train(pixels, targets)

            for i in xrange(len(shapes)):
                normalized_offset = regressor.apply(shapes[i], pixels[i])
                offset = util.transform_to_mean_shape(shapes[i], self.mean_shape).pseudoinverse().apply(normalized_offset).points
                shapes[i].points += offset

    def fit(self, image, initial_shape):
        image = image.as_greyscale()
        shape = fit_shape_to_box(initial_shape, get_bounding_box(image))

        for r in self.regressors:
            pixels = r.extract_features(image, shape)
            shape.points += util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse().apply(r.apply(shape, pixels)).points
            #shape.points += r.apply(shape, pixels).points

        return shape
