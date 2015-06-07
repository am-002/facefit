import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from menpo.visualize import print_dynamic
from menpofit.fittingresult import compute_error
import numpy as np
from esr import base
import util
import cv2
import menpodetect
from esr import second_level_cascade
from esr import fern_cascade
from esr import forest
from esr import fern

feature_extractor_builder = util.PixelExtractorBuilder(n_landmarks=68, n_pixels=400, kappa=0.3)

# primitive_regressor_builder = fern.FernBuilder(n_pixels=400, n_features=5, n_landmarks=68, beta=1000)
# primary_regressor_builder = fern_cascade.FernCascadeBuilder(primitive_regressor_builder, feature_extractor_builder,
#                                                            n_ferns=500, compress=False)

primitive_regressor_builder = forest.RegressionTreeBuilder(MU=1)
primary_regressor_builder = forest.RegressionForestBuilder(primitive_regressor_builder, feature_extractor_builder, n_trees=500)

builder = base.CascadedShapeRegressorBuilder(n_stages=10, n_perturbations=20, weak_builder=primary_regressor_builder)

# builder = base.CascadedShapeRegressorBuilder(n_stages=1, n_perturbations=1,
#                           weak_builder=forest.RegressionForestBuilder(n_trees=5, MU=1))

trainset = "/Users/andrejm/Google Drive/Work/BEng project/helen/subset"
testset = "/Users/andrejm/Google Drive/Work/BEng project/helen/subset"

face_detector = menpodetect.load_dlib_frontal_face_detector()

images = util.read_images(trainset, normalise=True)
gt_shapes = util.get_gt_shapes(images)
boxes = util.get_bounding_boxes(images, gt_shapes, face_detector)

model = builder.build(images, gt_shapes, boxes)

test_images = util.read_images(testset, normalise=True)

final_errors = []
for k, (im, gt_shape) in enumerate(zip(test_images, gt_shapes)):
    final_shapes = model.fit(im, util.get_bounding_boxes([im], [gt_shape], face_detector))
    final_shape = final_shapes[0]

    final_errors.append(compute_error(final_shape, gt_shape))

    print_dynamic('{}/{}'.format(k + 1, len(test_images)))

print '\nMean alignment error: ', np.mean(final_errors)

import hickle
hickle.dump(model, "blah.hkl")
