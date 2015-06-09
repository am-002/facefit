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
    ret = scipy_interpolation(im.pixels, points_to_sample, order=1, mode='constant', cval=0)

    return ret.reshape((len(points_to_sample),))

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
    x, y = box.points[0]
    w, h = box.range()

    # center_x = x + w*2.0/3.0
    # center_y = y + h/2.0

    center_x = x + w/2.0
    center_y = y + h/2.0

    shape = normal_shape.points - normal_shape.centre()
    shape *= [0.9*h/2.0, 0.9*w/2.0]
#     shape *= [h/2.0, w/2.0]
    shape += [center_x, center_y]

    return PointCloud(shape)


def center_shape(shape):
    return PointCloud(2 * (shape.points - shape.centre()) / shape.range())


def centered_mean_shape(target_shapes):
    mean_shape = menpo.shape.mean_pointcloud(target_shapes)
    return center_shape(mean_shape)

def perturb_shapes(shapes, gt_shapes, bboxes, n_perturbations):
    # indices = np.random.uniform(low=0, high=len(gt_shapes)-1, size=len(shapes))
    # ret = []
    # for i in xrange(len(shapes)):
    #     centered_shape = center_shape(gt_shapes[indices[i]])
    #     ret.append(fit_shape_to_box(centered_shape, bboxes[indices[i]]))
    # return ret

    dx = np.random.uniform(low=-0.15, high=0.15, size=(len(shapes)))
    dy = np.random.uniform(low=-0.15, high=0.15, size=(len(shapes)))
    # TODO: Rescale the mean shape as part of perturbations.
    # scale = np.random.normal(1, 0.12, size=(len(shapes)))
    scale = np.ones(len(shapes))
    normalized_offsets = np.dstack((dy, dx))[0]

    ret = []

    for i in xrange(len(shapes)):
        if i % n_perturbations == 0:
            ret.append(shapes[i])
        else:
            ret.append(PointCloud((shapes[i].points - shapes[i].centre())*scale[i] + gt_shapes[i].centre() + shapes[i].range() * normalized_offsets[i]))
    return ret

def is_point_within(pt, bounds):
    x, y = pt
    return bounds[0][0] <= x <= bounds[1][0] and bounds[0][1] <= y <= bounds[1][1]

def get_bounding_boxes(images, gt_shapes, face_detector):
    ret = []
    for i, (img, gt_shape) in enumerate(zip(images, gt_shapes)):
        print_dynamic("Detecting face {}/{}".format(i, len(images)))
        boxes = face_detector(img)
        # Some images contain multiple faces, but are annotated only once.
        # We only remember the box that contains the gt_shape.
        found = False
        for box in boxes:
            if is_point_within(gt_shape.centre(), box.bounds()):
                ret.append(box)
                found = True
                break
        if not found:
            ret.append(gt_shape.bounding_box())


    return np.array(ret)

#MAX_FACE_WIDTH = 500.0
def read_images(img_glob, normalise):
    # Read the training set into memory.
    images = []
    for img_orig in mio.import_images(img_glob, verbose=True, normalise=normalise):
        if not img_orig.has_landmarks:
            continue
        # Convert to greyscale and crop to landmarks.
        img = img_orig.as_greyscale(mode='average').crop_to_landmarks_proportion_inplace(0.5)
        #img = img.resize((MAX_FACE_WIDTH, img.shape[1]*(MAX_FACE_WIDTH/img.shape[0])))
        images.append(img)
    return np.array(images)


