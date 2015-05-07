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
        self.n_fern_features = n_fern_features
        self.n_perturbations = n_perturbations
        # TODO: In the paper, the kappa was set to 0.3*distance-between-eye-pupils-in-mean-shape.
        self.kappa = 0.3

    def train(self, img_glob):
        # Read the training set into memory.
        images = []
        for img_orig in mio.import_images(img_glob):
            if not img_orig.has_landmarks:
                continue
            # Convert to greyscale and crop to landmarks.
            images.append(img_orig.as_greyscale().crop_to_landmarks_proportion_inplace(0.5))

        # Calculate normalized mean shape centered at the origin.
        target_shapes = [img.landmarks['PTS'].lms for img in images]
        self.mean_shape = menpo.shape.mean_pointcloud(target_shapes)
        self.mean_shape.points = 2*(self.mean_shape.points - self.mean_shape.centre()) / self.mean_shape.range()

        n_images = len(images)
        n_landmarks = images[0].landmarks['PTS'].n_landmarks
        n_samples = n_images*self.n_perturbations

        self.regressors = [PrimaryRegressor(self.n_pixels, self.n_fern_features, self.n_ferns, n_landmarks, self.mean_shape, self.kappa)
                                for i in range(self.n_regressors)]

        targets = np.ndarray((n_samples, 2*n_landmarks))
        # shapes = np.ndarray((n_samples, n_landmarks, 2))
        pixels = np.ndarray((n_samples, self.n_pixels))

        # Generate initial shapes with perturbations.
        shapes = []
        for i in xrange(n_samples):
            bounding_box = get_bounding_box(images[i/self.n_perturbations])
            shapes.append(fit_shape_to_box(self.mean_shape, bounding_box))
            for j in xrange(self.n_perturbations-1):
                shapes.append(fit_shape_to_box(self.mean_shape, perturb(bounding_box)))

        # Perform regression in stages.
        for regressor_i, regressor in enumerate(self.regressors):
            for i, img in enumerate(images):
                for j in xrange(self.n_perturbations):
                    index = i*self.n_perturbations + j

                    delta = PointCloud(img.landmarks['PTS'].lms.points - shapes[index].points)
                    shape_to_mean = util.transform_to_mean_shape(shapes[index], self.mean_shape)
                    normalized_target = shape_to_mean.apply(delta)

                    targets[index] = normalized_target.as_vector()
                    mean_to_shape = shape_to_mean.pseudoinverse()
                    pixels[index] = regressor.extract_features(img, shapes[index], mean_to_shape)

            regressor.train(pixels, targets)

            for i in xrange(n_samples):
                normalized_offset = regressor.apply(pixels[i])
                offset = util.transform_to_mean_shape(shapes[i], self.mean_shape).pseudoinverse().apply(normalized_offset).points
                shapes[i].points += offset

    def fit(self, image, initial_shape):
        image = image.as_greyscale()
        shape = fit_shape_to_box(self.mean_shape, get_bounding_box(image))

        for r in self.regressors:
            mean_to_shape = util.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
            pixels = r.extract_features(image, shape, mean_to_shape)
            normalized_target = r.apply(pixels)
            offset =  mean_to_shape.apply(normalized_target).points
            shape.points += offset
            # shape.pomean_to_shapenverse().apply((shape, pixels)).points

        return shape
