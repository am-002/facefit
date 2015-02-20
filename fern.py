import random

INF = 102345678

class Bin:
    def __init__(self):
        self.size = 0.0
        self.offset_sum = [0, 0]

    def add(self, offset):
        self.size += 1.0
        self.offset_sum[0] += offset[0]
        self.offset_sum[1] += offset[1]

    def get_offset(self):
        if (self.size == 0.0):
            return [0, 0]
        return [self.offset_sum[0] / self.size, self.offset_sum[1] / self.size]

class Fern:
    def __init__(self, numOfFeatures):
        self.numOfFeatures = numOfFeatures
        self.thresholds = []
        self.bins = [None]*(2**numOfFeatures)
        for i in range(2**numOfFeatures):
            self.bins[i] = Bin()

    def getBin(self, features):
        res = 0
        for i in range(self.numOfFeatures):
            if features[i] <= self.thresholds[i]:
                res |= 1<<i
        return res

    # arr is an array of (features, offset) tuples.
    def train(self, arr):
        # Generate thersholds for each features. We first get the ranges
        # of all features in the training set.
        ranges = [[INF, -INF]]*(self.numOfFeatures)

        for (features, offset) in arr:
            for f in range(len(features)):
                ranges[f][0] = min(ranges[f][0], features[f])
                ranges[f][1] = max(ranges[f][1], features[f])

        self.thresholds = [0]*len(ranges)
        # Generate a random threshold for each features in the range.
        for f in range(len(ranges)):
            self.thresholds[f] = random.uniform(ranges[f][0], ranges[f][1])


        # Get the bin for each data point. In each one, store the average offset
        # of all landmarks in the bin.
        for (features, offset) in arr:
            ans = self.getBin(features)
            self.bins[ans].add(offset)

    def getOffset(self, features):
        b = self.getBin(features)
        return self.bins[b].get_offset()




