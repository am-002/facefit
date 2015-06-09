import sys

import cv2
import menpo
import hickle
import menpodetect


def add_landmarks(mat, shape):
    for i in xrange(18, 68):
        cv2.circle(mat, center=(int(shape.points[i][1]), int(shape.points[i][0])), radius=3, color=(0,255,0), thickness=-1)


cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 22)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

model = hickle.load(sys.argv[1], safe=False)
face_detector = menpodetect.load_dlib_frontal_face_detector()

last = False

#kernel = np.ones((3, 3), np.float32)/9

done = False

while True:
    ret, orig = cap.read()
    if not ret:
        continue
    frame=orig
#    frame = cv2.filter2D(orig, -1, kernel)
#    orig = frame

    orig_menpo = menpo.image.Image(orig).as_greyscale(mode='average')
    orig_menpo.pixels /= 255.0
    bbox = face_detector(orig_menpo)

    #if not last:
    img = menpo.image.Image(frame).as_greyscale(mode='average')

    #orig_shape = np.array(img.shape)
    #img = img.resize(orig_shape)

    shapes = model.apply(img,bbox)

    #else:
    #    last = False

    for shape in shapes:
        add_landmarks(orig, shape)
        for box in bbox:
            a, b = box.bounds()
            cv2.rectangle(orig, (a[1],a[0]), (b[1],b[0]), (0, 255, 0), 2)

    cv2.imshow('frame', orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
