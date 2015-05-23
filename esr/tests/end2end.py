import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from menpo.visualize import print_dynamic
from menpofit.fittingresult import compute_error
import numpy as np
from esr import base
import util
import cv2

builder = base.ESRBuilder()

trainset = "/Users/andrejm/Google Drive/Work/BEng project/helen/subset"
testset = "/Users/andrejm/Google Drive/Work/BEng project/helen/subset"

face_detector = cv2.CascadeClassifier("../../haarcascade_frontalface_default.xml")

images = util.read_images(trainset)
gt_shapes = util.get_gt_shapes(images)
boxes = util.get_bounding_boxes(images, face_detector)

model = builder.build(images, gt_shapes, boxes)

test_images = util.read_images(testset)

final_errors = []
for k, (im, gt_shape) in enumerate(zip(test_images, gt_shapes)):
    final_shape = model.fit(im, util.get_bounding_box(im, face_detector))

    final_errors.append(compute_error(final_shape, gt_shape))

    print_dynamic('{}/{}'.format(k + 1, len(test_images)))

print('Mean alignment error: ', np.mean(final_errors))

