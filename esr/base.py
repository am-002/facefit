import util
from util import *
from fern_cascade import FernCascadeBuilder

class ESRBuilder:
    def __init__(self, n_landmarks=68, n_stages=10, n_pixels=400, n_fern_features=5, n_ferns=500, n_perturbations=20,
                 kappa=0.3, beta=1000, stddev_perturb=0.04, basis_size=512, compression_maxnonzero=5, compress=True, weak_builder=None):
        self.n_landmarks = n_landmarks
        self.n_stages = n_stages
        self.n_pixels = n_pixels
        self.n_ferns = n_ferns
        self.n_fern_features = n_fern_features
        self.n_perturbations = n_perturbations
        self.kappa = kappa
        self.beta = beta
        self.mean_shape = None
        self.stddev_perturb = stddev_perturb
        self.basis_size = basis_size
        self.compression_maxnonzero = compression_maxnonzero
        self.compress = compress
        self.weak_builder = weak_builder
        if not weak_builder:
            self.weak_builder = FernCascadeBuilder(self.n_pixels, self.n_fern_features, self.n_ferns, self.n_landmarks,
                             kappa, self.beta, self.basis_size, self.compression_maxnonzero, self.compress)

    def build(self, images, gt_shapes, boxes):
        # images = np.array(self.read_images(images))
        self.mean_shape = util.centered_mean_shape(gt_shapes)

        # Generate initial shapes with perturbations.
        print_dynamic('Generating initial shapes')
        shapes = np.array([util.fit_shape_to_box(self.mean_shape, box) for box in boxes])

        print_dynamic('Perturbing initial estimates')
        boxes = boxes.repeat(self.n_perturbations, axis=0)
        images = images.repeat(self.n_perturbations, axis=0)
        gt_shapes = gt_shapes.repeat(self.n_perturbations, axis=0)
        shapes = shapes.repeat(self.n_perturbations, axis=0)

        if self.n_perturbations > 1:
            shapes = util.perturb_shapes(shapes, gt_shapes, boxes, self.n_perturbations)

        assert(len(boxes) == len(images))
        assert(len(shapes) == len(images))
        assert(len(gt_shapes) == len(images))

        print('\nSize of augmented dataset: {} images.\n'.format(len(images)))

        weak_regressors = []
        for j in xrange(self.n_stages):
            #KAPPA = self.kappa - j*0.023
            #fern_cascade_builder = FernCascadeBuilder(self.n_pixels, self.n_fern_features, self.n_ferns, self.n_landmarks,
            #                                          self.mean_shape, KAPPA, self.beta, self.basis_size, self.compression_maxnonzero, self.compress)
            #fern_cascade = fern_cascade_builder.build(images, shapes, gt_shapes, self.mean_shape)
            weak_regressor = self.weak_builder.build(images, shapes, gt_shapes, self.mean_shape)
            # Update current estimates of shapes.
            for i, (image, shape) in enumerate(zip(images, shapes)):
                offset = weak_regressor.apply(image, shape, transform_to_mean_shape(shape, self.mean_shape).pseudoinverse())
                shapes[i].points += offset.points
            weak_regressors.append(weak_regressor)
            print("\nBuilt outer regressor {}\n".format(j))

        return ESR(self.n_landmarks, weak_regressors, self.mean_shape)


class ESR:
    def __init__(self, n_landmarks, fern_cascades, mean_shape):
        self.n_landmarks = n_landmarks
        self.fern_cascades = fern_cascades
        self.mean_shape = mean_shape

    def fit(self, image, boxes):

        shapes = [fit_shape_to_box(self.mean_shape, box) for box in boxes]

        for shape in shapes:
            for r in self.fern_cascades:
                mean_to_shape = transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
                offset = r.apply(image, shape, mean_to_shape)
                shape.points += offset.points
        return shapes
