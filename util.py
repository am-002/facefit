import menpo
from menpo import io as mio
from menpo.shape import PointCloud
from menpo.transform import AlignmentSimilarity

__author__ = 'andrejm'

def translate_to_origin(src):
    center = src.centre()
    src = PointCloud([p - center for p in src.points])
    return src

def transform_to_mean_shape(src, mean_shape):
    src = translate_to_origin(src)

    return AlignmentSimilarity(src, mean_shape)


def getNormalisedMeanShape(img_path):
    mean_shape = menpo.shape.mean_pointcloud([img.landmarks['PTS'].lms for img in mio.import_images(img_path)])
    mean_shape = translate_to_origin(mean_shape)

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

    return PointCloud( [ (p[0]*w/2 + box_center[0], p[1]*h/2+box_center[1]) for p in normal_shape.points ])
    #return PointCloud([ [p[0]*w + box[0][0], p[1]*h+box[0][1]] for p in normal_shape.points])


def get_bounding_box(img):
    if not img.has_landmarks:
        return [[0,0], [0,0]]
    l = zip(*list(img.landmarks['PTS'].lms.points))
    box = [ [min(l[0]), min(l[1])], [max(l[0]), max(l[1])] ]
    return box