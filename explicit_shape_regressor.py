import sys

from menpo.shape import PointCloud
import menpo.io as mio

from primary_regressor import PrimaryRegressor
import util
from util import *
import copy


class ExplicitShapeRegressor:
    def __init__(self, n_landmarks, n_regressors, n_pixels, n_fern_features, n_ferns):
        self.n_landmarks = n_landmarks
        self.n_regressors = n_regressors
        self.n_pixels = n_pixels
        self.n_ferns = n_ferns

        # Calculate mean shape from a subset of training data.
        self.mean_shape = getNormalisedMeanShape('../helen/subset_cropped/')

        # TODO: In the paper, the kappa was set to 0.3*distance-between-eye-pupils-in-mean-shape.
        self.kappa = 0.3

        self.regressors = [PrimaryRegressor(n_pixels, n_fern_features, n_ferns, n_landmarks, self.mean_shape, self.kappa)
                                for i in range(n_regressors)]

    def train(self, img_glob):
        init_shape = self.mean_shape
        #n_samples = sum(1 for img in mio.import_images(img_glob) if img.has_landmarks)
        #shapes = [fit_shape_to_box(init_shape) for i in xrange(n_samples)]
        #shapes = [fit_shape_to_box(init_shape, get_bounding_box(img)) for img in mio.import_images(img_glob) if img.has_landmarks]
        #n_samples = len(shapes)
        shapes = []

        RESULTS = []

        for r in self.regressors:
            pixels = []
            targets = []
            sys.stdout.flush()
            n_samples = 0
            for img_i in mio.import_images(img_glob):
                if not img_i.has_landmarks:
                    continue
                # Convert to greyscale
                img = img_i.as_greyscale()
                if len(shapes) <= n_samples:
                    shapes.append(fit_shape_to_box(init_shape, get_bounding_box(img)))
                pixels.append(r.extract_features(img, shapes[n_samples]))
                delta = PointCloud(img.landmarks['PTS'].lms.points - shapes[n_samples].points)
                normalized_target = util.transform_to_mean_shape(shapes[n_samples], self.mean_shape).apply(delta)
                targets.append(normalized_target)
                n_samples += 1
                RESULTS.append([])

            r.train(pixels, targets)

            for i in xrange(n_samples):
                RESULTS[i].append(copy.deepcopy(shapes[i-1]))

            for i in xrange(n_samples):
                normalized_offset = r.apply(shapes[i], pixels[i])
                # print normalized_offset.points
                offset = util.transform_to_mean_shape(shapes[i], self.mean_shape).pseudoinverse().apply(normalized_offset).points
                shapes[i].points += offset
                RESULTS[i].append(copy.deepcopy(shapes[i-1]))
                #shapes[i].points += r.apply(shapes[i], pixels[i]).points

        return RESULTS

    def fit(self, image, initial_shape):
        image = image.as_greyscale()
        shape = fit_shape_to_box(initial_shape, get_bounding_box(image))

        for r in self.regressors:
            pixels = r.extract_features(image, shape)
            shape.points += util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse().apply(r.apply(shape, pixels)).points
            #shape.points += r.apply(shape, pixels).points

        return shape
