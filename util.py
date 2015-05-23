import menpo
from menpo import io as mio
from menpo.shape import PointCloud
from menpo.visualize import print_dynamic
import numpy as np
from menpo.transform import AlignmentSimilarity
from menpo.image.interpolation import scipy_interpolation

def sample_image(im, points_to_sample):
    if isinstance(points_to_sample, PointCloud):
            points_to_sample = points_to_sample.points
    return scipy_interpolation(im.pixels, points_to_sample, order=1, mode='constant', cval=0).reshape((len(points_to_sample),))

def transform_to_mean_shape(src, mean_shape):
    centered = PointCloud(src.points - src.centre(), copy=False)

    return AlignmentSimilarity(centered, mean_shape)

def normalize(ret):
    length = np.linalg.norm(ret)
    if length > 0:
        return ret / length
    return ret

def rand_unit_vector(dim):
    ret = np.random.randn(dim)
    return normalize(ret)

def get_gt_shapes(images):
    return np.array([image.landmarks['PTS'].lms for image in images])

def fit_shape_to_box(normal_shape, box):
    (x, y, w, h) = box

    center_x = x + w*2.0/3.0
    center_y = y + h/2.0

    shape = normal_shape.points - normal_shape.centre()
    shape *= [0.9*h/2.0, 0.9*w/2.0]
    shape += [center_x, center_y]

    return PointCloud(shape)

def centered_mean_shape(target_shapes):
    mean_shape = menpo.shape.mean_pointcloud(target_shapes)
    return PointCloud(2 * (mean_shape.points - mean_shape.centre()) / mean_shape.range())

def perturb_shapes(shapes):
    dx = np.random.uniform(low=-0.1, high=0.1, size=(len(shapes)))
    dy = np.random.uniform(low=-0.1, high=0.1, size=(len(shapes)))
    normalized_offsets = np.dstack((dy, dx))[0]

    ret = []
    for i in xrange(len(shapes)):
        ret.append(PointCloud(shapes[i].points + shapes[i].range() * normalized_offsets[i]))
    return ret

def get_bounding_box(image, face_detector):
    gray = image.pixels.astype(int)
    gray = gray.reshape((len(gray), len(gray[0]),))
    gray = np.array(gray, dtype=np.uint8)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    boxes = []
    maxi = 0
    for i, (x, y, w, h) in enumerate(faces):
        # We save the box in the menpo coordinate system.
        box = [y,x,h,w]
        boxes.append(box)
        if w > boxes[maxi][3]:
            maxi = i

    if len(faces) == 0:
        a, b = get_gt_shapes([image])[0].bounds()
        x, y = a
        w, h = b[0] - a[0], b[1] - a[1]
        return np.array([x-0.05*w, y-0.05*h, 1.05*w, 1.05*h])
    return boxes[maxi]

def read_images(img_glob):
    # Read the training set into memory.
    images = []
    for img_orig in mio.import_images(img_glob, verbose=True, normalise=False):
        if not img_orig.has_landmarks:
            continue
        # Convert to greyscale and crop to landmarks.
        images.append(img_orig.as_greyscale(mode='average').crop_to_landmarks_proportion_inplace(0.5))
    return np.array(images)

def get_bounding_boxes(images, face_detector):
    ret = []
    for i, image in enumerate(images):
        print_dynamic("Detecting face {}/{}".format(i, len(images)))
        ret.append(get_bounding_box(image, face_detector))
    return np.array(ret)


class FeatureExtractor:
    def __init__(self, n_landmarks, n_pixels, kappa):
        self.lmark = np.random.randint(low=0, high=n_landmarks, size=n_pixels)
        self.pixel_coords = np.random.uniform(low=-kappa, high=kappa, size=n_pixels*2).reshape(n_pixels, 2)

    def extract_features(self, img, shape, mean_to_shape):
        offsets = mean_to_shape.apply(self.pixel_coords)
        ret = shape.points[self.lmark] + offsets
        return sample_image(img, ret)