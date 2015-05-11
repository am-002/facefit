import hickle
import numpy as np
import menpo.io as mio
import menpo
from menpo.shape import PointCloud
from util import *
from fern_cascade import FernCascadeBuilder

def get_gt_shape(image):
    return image.landmarks['PTS'].lms

def get_bounding_box(image):
    return np.array(get_gt_shape(image).bounds())

class ESRBuilder:
    def __init__(self, n_landmarks=68, n_stages=10, n_pixels=400, n_fern_features=5,
                 n_ferns=500, n_perturbations=20, kappa=0.3, beta=1000):
        self.n_landmarks = n_landmarks
        self.n_stages = n_stages
        self.n_pixels = n_pixels
        self.n_ferns = n_ferns
        self.n_fern_features = n_fern_features
        self.n_perturbations = n_perturbations
        self.kappa = kappa
        self.beta = beta

    def read_images(self, img_glob):
        # Read the training set into memory.
        images = []
        for img_orig in mio.import_images(img_glob, verbose=True):
            if not img_orig.has_landmarks:
                continue
            # Convert to greyscale and crop to landmarks.
            images.append(img_orig.as_greyscale(mode='average').crop_to_landmarks_proportion_inplace(0.5))
        return images

    # Fit a shape into a box. The shape has to be normalised and centered around
    # the origin (inside [-1, -1], [1, 1]).
    @staticmethod
    def fit_shape_to_box(normal_shape, box):
        box = PointCloud(box, copy=False)
        return PointCloud(normal_shape.points*box.range()/2 + box.centre())

    def from_file(self, file):
        return hickle.load(file, safe=False)

    def centered_mean_shape(self, target_shapes):
        mean_shape = menpo.shape.mean_pointcloud(target_shapes)
        return PointCloud(2 * (mean_shape.points - mean_shape.centre()) / mean_shape.range())

    def perturb_boxes(self, boxes, n_perturbations):
        perturbed_positions = np.random.normal(loc=1, scale=0.1, size=(len(boxes)*n_perturbations, 2, 2))
        return boxes.repeat(n_perturbations, axis=0) * perturbed_positions

    def get_gt_shapes(self, images):
        return [img.landmarks['PTS'].lms for img in images]

    def get_bounding_boxes(self, images):
        return np.array([shape.bounds() for shape in self.get_gt_shapes(images)])

    def build(self, images):
        images = np.array(images)
        self.mean_shape = self.centered_mean_shape([img.landmarks['PTS'].lms for img in images])

        # Generate initial shapes with perturbations.
        shapes = [self.fit_shape_to_box(self.mean_shape, box) for box in
                        self.perturb_boxes(self.get_bounding_boxes(images), self.n_perturbations)]

        # Repeat each image n_perturbations times. Only shallow-duplicates references.
        images = images.repeat(self.n_perturbations)

        # Extract ground truth shapes from annotated images.
        gt_shapes = self.get_gt_shapes(images)

        fern_cascades = []
        for i in xrange(self.n_stages):
            fern_cascade_builder = FernCascadeBuilder(self.n_pixels, self.n_fern_features, self.n_ferns,
                                                      self.n_landmarks, self.mean_shape, self.kappa, self.beta)
            fern_cascade = fern_cascade_builder.build(images, shapes, gt_shapes)
            # Update current estimates of shapes.
            shapes = [fern_cascade.apply(image, shape, transform_to_mean_shape(shape, self.mean_shape))
                      for image, shape in zip(images, shapes)]
            fern_cascades.append(fern_cascade)

        return ESR(self.n_landmarks, fern_cascades, self.mean_shape)

class ESR:
    def __init__(self, n_landmarks, fern_cascades, mean_shape):
        self.n_landmarks = n_landmarks
        self.fern_cascades = fern_cascades
        self.mean_shape = mean_shape

    def fit(self, image, initial_shape):
        assert(initial_shape.n_points == self.n_landmarks)
        image = image.as_greyscale(mode='average')
        shape = ESRBuilder.fit_shape_to_box(initial_shape, get_bounding_box(image))

        for r in self.fern_cascades:
            mean_to_shape = transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
            normalized_target = r.apply(image, shape, mean_to_shape)
            shape.points += mean_to_shape.apply(normalized_target).points

        return shape
