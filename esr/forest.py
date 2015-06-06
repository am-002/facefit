from menpo.shape import PointCloud
from menpo.visualize import print_dynamic

__author__ = 'andrejm'

import numpy as np
import util

class RegressionForest:
    def __init__(self, pixel_extractor, trees, n_landmarks):
        self.pixel_extractor = pixel_extractor
        self.trees = trees
        self.n_landmarks = n_landmarks

    def apply(self, image, shape, mean_to_shape):
        shape_indexed_features = self.pixel_extractor.extract_features(image, shape, mean_to_shape)
        res = PointCloud(np.zeros((self.n_landmarks, 2)), copy=False)
        for r in self.trees:
            offset = r.apply(shape_indexed_features)
            # print "Forest returned {}".format(offset)
            res.points += offset.reshape((self.n_landmarks, 2))
        return mean_to_shape.apply(res)

class RegressionForestBuilder:
    def __init__(self, n_trees):
        self.mean_shape = None
        self.n_trees = n_trees

    def to_mean(self, shape):
        return util.transform_to_mean_shape(shape, self.mean_shape)

    def build(self, images, shapes, gt_shapes, mean_shape):
        self.mean_shape = mean_shape
        #TODO
        self.n_landmarks = 68
        #self.n_trees = 500
        self.n_pixels = 400
        pixel_extractor = util.FeatureExtractor(self.n_landmarks, self.n_pixels, 0.3)

        # Calculate normalized targets.
        deltas = [gt_shape.points - shape.points for gt_shape, shape in zip(gt_shapes, shapes)]
        targets = np.array([self.to_mean(shape).apply(delta) for (shape, delta) in zip(shapes, deltas)])

        #print "Deltas:"
        #print deltas

        # Extract shape-indexed pixels from images.
        pixel_vectors = np.array([pixel_extractor.extract_features(img, shape, self.to_mean(shape).pseudoinverse())
                                  for (img, shape) in zip(images, shapes)])

        pixel_mean_coords = self.mean_shape.points[pixel_extractor.lmark] + pixel_extractor.pixel_coords

        targets = targets.reshape((len(targets), self.n_landmarks*2))

        trees = []
        for i in xrange(self.n_trees):
            print_dynamic("Building regression tree {}".format(i))
            tree_builder = RegressionTreeBuilder()
            tree = tree_builder.build(pixel_vectors, targets, pixel_mean_coords)
            # Update targets.
            targets -= [tree.apply(pixel_vector) for pixel_vector in pixel_vectors]
            trees.append(tree)
        return RegressionForest(pixel_extractor, trees, self.n_landmarks)

class RegressionTree:
    def __init__(self, splits, leaves, depth=5):
        self.splits = splits
        self.leaves = leaves
        self.depth = depth

    def apply(self, pixels):
        # print 'Applying on '
        # print pixels
        node = 0
        for k in xrange(self.depth-1):
            i, j, thresh = self.splits[node]
            node = 2*node+1
            if pixels[i] - pixels[j] > thresh:
                node += 1
        # print 'Fell into node {}'.format(node)
        return self.leaves[node - len(self.splits)]

class RegressionTreeBuilder:
    def __init__(self, n_landmarks=68, depth=5, n_test_features=20):
        self.depth = depth
        self.n_landmarks = n_landmarks
        self.n_test_splits = n_test_features
        self.n_split_nodes = (1 << (depth-1)) - 1

    def build(self, pixels, targets, pixel_mean_coords):
        buckets = [None for _ in xrange((1 << self.depth) - 1)]
        sums = [0 for _ in xrange((1 << self.depth) - 1)]
        cnts = [0 for _ in xrange((1 << self.depth) - 1)]

        splits = []

        buckets[0] = (0, len(targets))
        sums[0] = targets.sum(axis=0)
        cnts[0] = len(targets)
        perm = np.arange(0, len(targets), dtype=int)
        #print sums[0].shape

        for i in xrange(self.n_split_nodes):
            split, division, best_sums = self.get_best_split(pixels, targets, perm, buckets[i][0], buckets[i][1],
                                             pixel_mean_coords, sums[i], cnts[i], self.n_test_splits)
            begin, mid, end = division
            splits.append(split)
            #print "At node {}, dividing {} datapoints to buckets of size {} and {}\n".format(i,buckets[i][1]-buckets[i][0], mid-begin, end-mid)
            buckets[2*i+1] = (begin, mid)
            buckets[2*i+2] = (mid, end)
            sums[2*i+1], sums[2*i+2] = best_sums
            cnts[2*i+1] = (mid-begin)
            cnts[2*i+2] = (end-mid)

        leaves = np.zeros(shape=(1 << (self.depth-1), self.n_landmarks*2))


        for i in xrange(self.n_split_nodes, (1 << self.depth) - 1):
            if cnts[i] != 0:
                #TODO
                MU = 0.1
                #MU = 1
                leaves[i - self.n_split_nodes] = MU*sums[i] / cnts[i]
                s = ""
                for k in xrange(int(buckets[i][0]), int(buckets[i][1])):
                    s += " " + str(perm[k])
                # print "Putting {} into node {}".format(s, i)
        return RegressionTree(splits, leaves, self.depth)

    # Generate a random split w.r.t an exponential prior.
    # Takes coordinates in the "mean shape space".
    def gen_random_split(self, mean_coords, LAMBDA=0.1):
        i, j = 0, 0
        while True:
            i, j = np.random.randint(low=0, high=len(mean_coords), size=2)
            dist = abs(np.linalg.norm(mean_coords[i] - mean_coords[j]))
            prob = np.exp(-dist/LAMBDA)
            if i != j and prob > np.random.random():
                break
        # TODO: Assuming normalised images.
        threshold = np.random.uniform(low=-0.25, high=0.25)
        return int(i), int(j), threshold

    def get_best_split(self, pixels, targets, perm, begin, end, pixel_mean_coords, overall_sum, overall_cnt, n_test_splits = 20):
        splits = np.array([self.gen_random_split(pixel_mean_coords) for _ in xrange(n_test_splits)])
        pix1 = np.array(splits[:, 0], dtype=int)
        pix2 = np.array(splits[:, 1], dtype=int)
        divisions = (pixels[perm[begin:end]][:, pix1] - pixels[perm[begin:end]][:, pix2] > splits[:, 2]).transpose()

        best_div_score = -1
        best_division_index = 0
        best_midpoint = begin
        # TODO
        best_sums = (np.zeros(136), np.zeros(136))


        for i, division in enumerate(divisions):
            right_sum = targets[perm[begin:end]][division].sum(axis=0)
            right_cnt = float(np.count_nonzero(division))

            #print overall_sum
            #print right_sum
            #print 'begin, end : {}{}'.format(begin, end)
            #print overall_sum.shape
            #print right_sum.shape
            left_sum = overall_sum - right_sum
            left_cnt = overall_cnt - right_cnt
            #if left_cnt == 0 or right_cnt == 0:
            #    continue
            # TODO: Should this be *left_cnt, *right_cnt?
            lcnt = left_cnt
            rcnt = right_cnt
            if right_cnt == 0:
                rcnt = 1
            if left_cnt == 0:
                lcnt = 1
            score = left_sum.dot(left_sum)/lcnt + right_sum.dot(right_sum)/rcnt
            if score > best_div_score:
                best_division_index = i
                #right_targets = targets[np.invert(division)]
                #best_division = (left_targets, right_targets)
                best_midpoint = begin + left_cnt
                best_div_score = score
                best_sums = (left_sum, right_sum)

        ind = np.argsort(divisions[best_division_index])
        perm[begin:end] = perm[begin:end][ind]

        #interval_left = (begin, best_midpoint)
        #interval_right = (best_midpoint, end)

        return (splits[best_division_index], (begin, best_midpoint, end), best_sums)