import cv2
import menpo
import menpo.io as mio
import numpy as np
import hickle

def get_img():
    ret, frame = cap.read()
    if not ret:
        return None
    img = menpo.image.Image(frame).as_greyscale(mode='average')
    img.resize((640, 480))
    return img

def menpo2cv(img):
    img = img.resize((360,640))
    mat = img.pixels.reshape((360,640))
    mat = np.array(mat, dtype=np.uint8)
    return mat

def add_landmarks(mat, shape):
    for i in xrange(68):
        cv2.circle(mat, center=(int(shape.points[i][1]), int(shape.points[i][0])), radius=2, color=255, thickness=-1)


cap = cv2.VideoCapture(0)
model = hickle.load("model_new.hkl", safe=False)
model.face_detector = cv2.CascadeClassifier("../haarcascade_frontalface_alt.xml")

while True:
    img = get_img()
    img = img.resize((360, 640))
    try:
        shape = model.fit(img, model.mean_shape)
        mat = menpo2cv(img)
        add_landmarks(mat, shape)
        cv2.imshow('frame', mat)
    except:
        cv2.imshow('frame', np.array(img.pixels.reshape((360,640,)), dtype=np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
