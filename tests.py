from explicit_shape_regressor import *
from menpo.shape import PointCloud
from primitive_regressor import Fern


# def test_fern():
#     f = Fern(n_features = 7, n_landmarks = 2)
#     features = [
#         [1, 0.3, 20, 10, 1.3, -1, -5],
#         [6, -0.9, 20, 10, 2.3, -1, -4],
#         [4, -46, 20, 10, -11.3, -1, -3],
#         [12, -3, 3, 2.1, 111.3, -11.31, -44]
#     ]
#     targets = [
#         PointCloud([1, 2]),
#         PointCloud([-1.2, 4.3]),
#         PointCloud([2, -3.2]),
#         PointCloud([3.2, -1.2])
#     ]
#     f.train(features, targets)
#
#     # Check if fern's results are linear combinations of targets.
#
#     for feature_vector in features:
#         f.test(feature_vector)

def test_end2end():
    esr = ExplicitShapeRegressor(nLandmarks=68, nRegressors=1, P=5, nFernFeatures=1, nFerns=1)
    esr.train("../helen/subset/*")
    img = mio.import_image('../helen/one/3266693323_1.jpg')
    #print img.landmarks['PTS'].lms.points
    pc = esr.fit(img, esr.mean_shape)
    #print pc.points


# test_fern()
test_end2end()