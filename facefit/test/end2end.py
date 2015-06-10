import unittest
import sys
import os
import menpo.io as mio
from menpo.visualize import print_dynamic
from menpofit.fittingresult import compute_error
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from facefit import util
from facefit import esr, ert, lbf
import menpodetect

def test_model(model, test_images, num_init):
    face_detector = menpodetect.load_dlib_frontal_face_detector()
    test_gt_shapes = util.get_gt_shapes(test_images)
    test_boxes = util.get_bounding_boxes(test_images, test_gt_shapes, face_detector)

    initial_errors = []
    final_errors = []

    initial_shapes = []
    final_shapes = []

    for k, (im, gt_shape, box) in enumerate(zip(test_images, test_gt_shapes, test_boxes)):
        init_shapes, fin_shapes = model.apply(im, ([box], num_init, None))

        init_shape = util.get_median_shape(init_shapes)
        final_shape = fin_shapes[0]

        initial_shapes.append(init_shape)
        final_shapes.append(final_shape)

        initial_errors.append(compute_error(init_shape, gt_shape))
        final_errors.append(compute_error(final_shape, gt_shape))

        print_dynamic('{}/{}'.format(k + 1, len(test_images)))

    return initial_errors, final_errors, initial_shapes, final_shapes

def fit_all(model_builder, train_images, test_images, num_init):
    face_detector = menpodetect.load_dlib_frontal_face_detector()

    train_gt_shapes = util.get_gt_shapes(train_images)
    train_boxes = util.get_bounding_boxes(train_images, train_gt_shapes, face_detector)

    model = model_builder.build(train_images, train_gt_shapes, train_boxes)

    initial_errors, final_errors, initial_shapes, final_shapes = test_model(model, test_images, num_init)

    return initial_errors, final_errors, initial_shapes, final_shapes, model


# Use a handful of images built into menpo for both training and testing.
images = ['einstein.jpg', 'takeo.ppm']
test_images = np.array([mio.import_builtin_asset(image).as_greyscale(mode='average').crop_to_landmarks_proportion_inplace(0.5)
                        for image in images])
train_images = test_images

def test_all(test, model_builder, test_images, train_images):
    initerr, finerr, _, _, _ = fit_all(model_builder, test_images, train_images, num_init=1)

    init_mean_error = np.mean(initerr)
    fin_mean_error = np.mean(finerr)

    print "Mean initial error: {}\n".format(init_mean_error)
    print "Mean final error: {}\n".format(fin_mean_error)

    test.failIfAlmostEqual(fin_mean_error, init_mean_error/10.0)
    test.failUnlessAlmostEqual(fin_mean_error, 0, places=4)


class ESRTest(unittest.TestCase):
    def test_end2end(self):
        esr_builder = esr.ESRBuilder(n_landmarks=68, n_stages=1, n_ferns=1, beta=0, n_perturbations=1, compress_ferns=False)
        test_all(self, esr_builder, test_images, train_images)


class ERTTest(unittest.TestCase):
    def test_end2end(self):
        ert_builder = ert.ERTBuilder(n_stages=1, n_trees=1, MU=1, n_perturbations=1)
        test_all(self, ert_builder, test_images, train_images)


class LBFTest(unittest.TestCase):
    def test_end2end(self):
        lbf_builder = lbf.LBFBuilder(n_stages=2, n_trees=68, tree_depth=5, n_perturbations=1, MU=1)
        test_all(self, lbf_builder, test_images, train_images)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
