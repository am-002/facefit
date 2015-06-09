import cv2
import menpo
import numpy as np
import hickle
import menpodetect
import sys
def add_landmarks(mat, shape):
    for i in xrange(68):
        cv2.circle(mat, center=(int(shape.points[i][1]), int(shape.points[i][0])), radius=2, color=255, thickness=-1)


cap = cv2.VideoCapture(0)
model = hickle.load(sys.argv[1], safe=False)
face_detector = menpodetect.load_dlib_frontal_face_detector()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = menpo.image.Image(frame).as_greyscale(mode='average')
    img.pixels /= 255.0

    orig_shape = np.array(img.shape)
    img = img.resize(orig_shape)

    shapes = model.fit(img, face_detector(img))

    for shape in shapes:
        add_landmarks(frame, shape)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
