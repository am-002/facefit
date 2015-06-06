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
from esr import forest

builder = base.ESRBuilder(n_stages=1, n_perturbations=1, weak_builder=None, beta=0, n_ferns=10, compress=False)

trainset = "/Users/andrejm/Google Drive/Work/BEng project/helen/two"
testset = "/Users/andrejm/Google Drive/Work/BEng project/helen/two"

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
    #print gt_shape.points
    #print final_shape.points

    final_errors.append(compute_error(final_shape, gt_shape))

    print_dynamic('{}/{}'.format(k + 1, len(test_images)))

print '\nMean alignment error: ', np.mean(final_errors)

import hickle
hickle.dump(model, "blah.hkl")
