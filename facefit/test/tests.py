import unittest

from facefit.esr.fern import *
from facefit.esr.fern import Fern
from facefit.util import *


def compare_arrays(self, a, b):
    return self.failUnless(np.isclose(np.array(a), b, atol=1e-5).all())

# class ESRTest(unittest.TestCase):
#     def test_fit_shape_to_box(self):
#         shapes = [
#             [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
#             [[0.0, 0.0]],
#         ]
#         boxes = [
#             [[10.0, 0.0], [15.0, 8.0]],
#             [[1.0, 1.0], [2.0, 2.0]],
#         ]
#         targets = [
#             [[10.0, 0.0], [10.0, 8.0], [15.0, 8.0], [15.0, 0.0]],
#             [[1.5, 1.5]],
#         ]
#
#         for shape, box, res in zip(shapes, boxes, targets):
#             ans = ESRBuilder.fit_shape_to_box(PointCloud(shape), np.array(box, dtype=float))
#             self.failUnless(np.all(ans.points == np.array(res)))


class FernBuilderTest(unittest.TestCase):
    def test_get_features(self):
        pixel_samples = [
            [[1.1, 2.3, 3.1, 4.42, 5.1, 5.2], [0.1, 2.3, 4.2, 0.14, 42.47, 42.42]],
        ]
        feature_indices = [
            [[1, 2], [2, 3], [5, 4]]
        ]
        result = [
            [[-0.8, -1.32, 0.1], [-1.9, 4.06, -0.05]],
        ]

        for pixels, feat_ind, res in zip(pixel_samples, feature_indices, result):
            ans = FernBuilder._get_features(np.array(pixels), np.array(feat_ind))
            self.failUnless(np.isclose(np.array(res), ans).all())


    def test_calc_bin_averages(self):
        test_cases = [
            {
                "n_landmarks": 2,
                "n_features": 2,
                "beta": 1000,
                "targets": [[1, 2.0, 3, 4],
                            [0, 1.2, 44.1, 1],
                            [1.4, 141.4, 55, 2]],
                "bin_ids": [0, 1, 2],
                "result": [[0.000999, 0.001998, 0.002997, 0.003996],
                            [0, 0.0011988, 0.044055944, 0.000999],
                            [0.0013986, 0.1412587412, 0.05494505495, 0.001998],
                            [0, 0, 0, 0]]
            }
        ]

        for t in test_cases:
            ans = FernBuilder._calc_bin_averages(np.array(t['targets']),
                                                 np.array(t['bin_ids']),
                                                 t['n_features'], t['n_landmarks'],
                                                 t['beta'])
            res = np.array(t["result"])
            compare_arrays(self, ans, np.array(res))

class FernTest(unittest.TestCase):
    def setUp(self):
        features = np.array([ [0, 1], [1, 2] ], dtype=int)
        bins = [
           [1, 2, 3, 4],
           [0, 0, 1, 1],
           [34, 33, 13, 134],
           [52, 42, 57, 21]
        ]
        thresholds = [ 10, 5]
        self.fern1 = Fern(2, features, np.array(bins), np.array(thresholds))

    def test_apply(self):
        test_cases = [
            {
                "pixels": [100, 14, 123],
                "result": [[34, 33], [13, 134]]
            }
        ]

        for t in test_cases:
            ans = self.fern1.apply(np.array(t['pixels']))
            compare_arrays(self, ans, np.array(t['result']))

    # def test_get_bin_ids(self):
    #     features = [
    #         [ [1, 2, 3], [4, 2, 1] ],
    #     ]
    #     thresholds = [
    #         [1, 0, 4],
    #     ]
    #     result = [
    #         [5, 4],
    #     ]
    #
    #     for f, t, r in zip(features, thresholds, result):
    #         ans = Fern._get_bin_ids(np.array(f), np.array(t))
    #         self.failIf(np.any(ans != np.array(r)))


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


def main():
    unittest.main()

if __name__ == '__main__':
    main()