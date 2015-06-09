import numpy as np
from facefit.base import Regressor, RegressorBuilder


class RegressionTreeBuilder(RegressorBuilder):
    def __init__(self, depth, n_test_features, exponential_prior, MU):
        self.depth = depth
        self.n_test_splits = n_test_features
        self.n_split_nodes = (1 << (depth-1)) - 1
        self.MU = MU
        self.exponential_prior = exponential_prior

    def build(self, pixels, targets, data):
        pixel_mean_coords, mean_shape = data
        buckets = [None for _ in xrange((1 << self.depth) - 1)]
        sums = [0 for _ in xrange((1 << self.depth) - 1)]
        cnts = [0 for _ in xrange((1 << self.depth) - 1)]

        splits = []

        buckets[0] = (0, len(targets))
        sums[0] = targets.sum(axis=0)
        cnts[0] = len(targets)
        target_size = len(targets[0])
        perm = np.arange(0, len(targets), dtype=int)

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
        leaves = np.zeros(shape=(1 << (self.depth-1), target_size))

        for i in xrange(self.n_split_nodes, (1 << self.depth) - 1):
            if cnts[i] != 0:
                leaves[i - self.n_split_nodes] = self.MU*sums[i] / cnts[i]
                s = ""
                for k in xrange(int(buckets[i][0]), int(buckets[i][1])):
                    s += " " + str(perm[k])
        return RegressionTree(splits, leaves, self.depth)

    # Generate a random split w.r.t an exponential prior.
    # Takes coordinates in the "mean shape space".
    def gen_random_split(self, mean_coords, LAMBDA=0.1):
        i, j = 0, 0

        while True:
            i, j = np.random.randint(low=0, high=len(mean_coords), size=2)
            dist = abs(np.linalg.norm(mean_coords[i] - mean_coords[j]))
            prob = np.exp(-dist/LAMBDA)
            if i != j and prob > np.random.random() or not self.exponential_prior:
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
        target_size = len(targets[0])
        best_sums = (np.zeros(target_size), np.zeros(target_size))

        for i, division in enumerate(divisions):
            right_sum = targets[perm[begin:end]][division].sum(axis=0)
            right_cnt = float(np.count_nonzero(division))

            left_sum = overall_sum - right_sum
            left_cnt = overall_cnt - right_cnt
            # TODO: Should this be *left_cnt, *right_cnt?
            lcnt = left_cnt
            rcnt = right_cnt
            if right_cnt == 0:
                rcnt = 1
            if left_cnt == 0:
                lcnt = 1
            score = left_sum.dot(left_sum)/lcnt + right_sum.dot(right_sum)/rcnt
            # print score
            if score > best_div_score:
                best_division_index = i
                best_midpoint = begin + left_cnt
                best_div_score = score
                best_sums = (left_sum, right_sum)

        ind = np.argsort(divisions[best_division_index])
        perm[begin:end] = perm[begin:end][ind]

        return (splits[best_division_index], (begin, best_midpoint, end), best_sums)


class RegressionTree(Regressor):
    def __init__(self, splits, leaves, depth):
        self.splits = splits
        self.leaves = leaves
        self.depth = depth

    def get_leaf_index(self, pixels, extra=None):
        node = 0
        for k in xrange(self.depth-1):
            i, j, thresh = self.splits[node]
            node = 2*node+1
            if pixels[i] - pixels[j] > thresh:
                node += 1
        return node - len(self.splits)

    def apply(self, pixels, extra=None):
        return self.leaves[self.get_leaf_index(pixels, extra)]