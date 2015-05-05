import menpo
import random
from menpo import io as mio
from menpo.shape import PointCloud
from menpo.transform import AlignmentSimilarity

__author__ = 'andrejm'

def translate_to_origin(src):
    src = PointCloud(src)
    center = src.centre()
    src.points -= center
    #src = PointCloud([p - center for p in src.points])
    return src

def transform_to_mean_shape(src, mean_shape):
    src = translate_to_origin(src)

    return AlignmentSimilarity(src, mean_shape)



def getNormalisedMeanShape(img_path):
    mean_shape = menpo.shape.mean_pointcloud([img.landmarks['PTS'].lms for img in mio.import_images(img_path)])
    mean_shape = translate_to_origin(mean_shape.points)

    l = zip(*list(mean_shape.points))
    box = [ [min(l[0]), min(l[1])], [max(l[0]), max(l[1])] ]
    w = box[1][0] - box[0][0]
    h = box[1][1] - box[0][1]

    return PointCloud([(p[0]/(w/2), p[1]/(h/2)) for p in mean_shape.points])


def fit_shape_to_box(normal_shape, box):
    w = box[1][0] - box[0][0]
    h = box[1][1] - box[0][1]
    box_center = [0, 0]
    box_center[0] = (box[0][0] + box[1][0])/2.0
    box_center[1] = (box[0][1] + box[1][1])/2.0

    # TODO: Slow.
    return PointCloud( [ (p[0]*w/2 + box_center[0], p[1]*h/2+box_center[1]) for p in normal_shape.points ])
    #return PointCloud([ [p[0]*w + box[0][0], p[1]*h+box[0][1]] for p in normal_shape.points])


# TODO: Use OpenCV for this!
def get_bounding_box(img):
    if not img.has_landmarks:
        return [[0,0], [0,0]]
    l = zip(*list(img.landmarks['PTS'].lms.points))
    box = [ [min(l[0]), min(l[1])], [max(l[0]), max(l[1])] ]

    # image.landmarks['PTS'].lms.bounds()!!

    return box

def perturb(bounding_box):
    width = bounding_box[1][0] - bounding_box[0][0]
    height = bounding_box[1][1] - bounding_box[0][1]

    for i in xrange(2):
        dx = random.uniform(-0.1*width, 0.1*width)
        dy = random.uniform(-0.1*height, 0.1*height)

        bounding_box[i] += [dx, dy]
    return bounding_box
