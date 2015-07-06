import sys
import site
import numpy as np
from os import path

dirname = path.dirname(path.abspath(__file__))
site.addsitedir(path.join(dirname, '..'))
from facefit.ert.tree import RegressionTree

import cv2
import menpo
import hickle
import menpodetect

def add_landmarks(mat, shape):
    for i in xrange(0, 68):
        cv2.circle(mat, center=(int(shape.points[i][1]), int(shape.points[i][0])), radius=3, color=(0,255,0), thickness=-1)

model = hickle.load(sys.argv[1], safe=False)
face_detector = menpodetect.load_dlib_frontal_face_detector()

WIDTH=640
HEIGHT=480

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Num of perturbations of the initial shape within a bounding box.
n_inits = 1

ret, orig = cap.read()
orig_menpo = menpo.image.Image(orig.mean(axis=2)/255.0)

while True:
    _, orig = cap.read()
    orig_menpo.pixels[:] = (orig.mean(axis=2)/255.0).reshape(HEIGHT, WIDTH, 1)
    bbox = face_detector(orig_menpo)
    _, shapes = model.apply(orig_menpo,(bbox, int(n_inits), None))

    for shape in shapes:
        add_landmarks(orig, shape)
        # Add bounding box around the face.
        # for box in bbox:
        #    a, b = box.bounds()
        #    cv2.rectangle(orig, (a[1],a[0]), (b[1],b[0]), (0, 255, 0), 2)

    cv2.imshow('frame', orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
