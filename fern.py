import random
import numpy
import menpo
from menpo.shape import PointCloud

INF = 102345678

class Bin:
    def __init__(self, nLandmarks):
        self.size = 0.0
        self.delta_sum = menpo.shape.PointCloud([[0,0]]*nLandmarks)

    def add(self, delta):
        self.size += 1.0
        self.delta_sum.points += delta.points

    def get_delta(self):
        if (self.size == 0.0):
            return self.delta_sum
        return PointCloud(self.delta_sum.points / self.size)

class Fern:
    def __init__(self, numOfFeatures, nLandmarks):
        self.numOfFeatures = numOfFeatures
        self.thresholds = []
        self.bins = [None]*(2**numOfFeatures)
        for i in range(2**numOfFeatures):
            self.bins[i] = Bin(nLandmarks)

    def getBin(self, features):
        res = 0
        for i in range(self.numOfFeatures):
            if features[i] <= self.thresholds[i]:
                res |= 1<<i
        return res

    # arr is an array of (features, delta) tuples.
    def train(self, arr):
        # Generate thersholds for each features. We first get the ranges
        # of all features in the training set.
        ranges = [[INF, -INF]]*(self.numOfFeatures)

        for (features, delta) in arr:
            for f in range(len(features)):
                ranges[f][0] = min(ranges[f][0], features[f])
                ranges[f][1] = max(ranges[f][1], features[f])

        self.thresholds = [0]*self.numOfFeatures
        # Generate a random threshold for each features in the range.
        for f in range(self.numOfFeatures):
            self.thresholds[f] = random.uniform(ranges[f][0], ranges[f][1])

        # Get the bin for each data point. In each one, store the average delta
        # of all landmarks in the bin.
        for (features, delta) in arr:
            ans = self.getBin(features)
            self.bins[ans].add(delta)

    def test(self, features):
        b = self.getBin(features)
        return self.bins[b].get_delta()
