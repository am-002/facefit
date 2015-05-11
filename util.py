import menpo
from menpo import io as mio
from menpo.shape import PointCloud
import numpy as np
from menpo.transform import AlignmentSimilarity


__author__ = 'andrejm'

from menpo.image.interpolation import scipy_interpolation

def sample_image(im, points_to_sample):
    if isinstance(points_to_sample, PointCloud):
            points_to_sample = points_to_sample.points
    return scipy_interpolation(im.pixels, points_to_sample, order=1,  mode='constant', cval=0).reshape((len(points_to_sample),))

def transform_to_mean_shape(src, mean_shape):
    centered = PointCloud(src.points - src.centre(), copy=False)

    return AlignmentSimilarity(centered, mean_shape)

def rand_unit_vector(dim):
    ret = np.random.randn(dim)
    ret /= np.linalg.norm(ret)
    return ret

# TODO: Use OpenCV for this!
def get_bounding_box(img):
    if not img.has_landmarks:
        return [[0,0], [0,0]]
    l = zip(*list(img.landmarks['PTS'].lms.points))
    box = [ [min(l[0]), min(l[1])], [max(l[0]), max(l[1])] ]

    # image.landmarks['PTS'].lms.bounds()!!

    return box


