# Correlation-based feature selection.

import numpy
import random
import math

class FeatureSelector:
    def __init__(self, P, F):
        self.P2 = P
        self.F = F
        self.features = [0]*F
        random.seed(None)

    def getPearson(self, tuples):
        x = []
        y = []
        for (a, b) in tuples:
            x.append(a)
            y.append(b)
        return numpy.corrcoef(x, y)[0, 1]


    def getRandomDirection(self):
        return [random.random()*10, random.random()*10]

    # Project a onto b and return length of the projection.
    def project(self, a, b):
        return (a[0]*b[0] + a[1]*b[1]) / math.sqrt(a[0]**2 + a[1]**2)


    def train(self, arr):
        for i in range(self.F):
            # Project the offset in random direction and measure the scalar length
            # of the projection. Find which feature has highest imapct on the length.

            d = self.getRandomDirection()
            featureVsLength = [None]*self.P2
            for j in range(self.P2):
                featureVsLength[j] = []

            for (originalFeatures, offset) in arr:
                l = self.project(offset, d)
                for origFeature in range(self.P2):
                    f = originalFeatures[origFeature]
                    featureVsLength[origFeature].append([f, l])


            maxim = -123456.0
            res = 0.0
            for origFeature in range(self.P2):
                corr = self.getPearson(featureVsLength[origFeature])
                if corr > maxim:
                    res = origFeature
                    maxim = corr
            self.features[i] = res
            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).
        # Here the features array is populated

    def selectFeatures(self, original):
        res = [0]*F
        for i in range(F):
            res[i] = original[self.features[i]]
        return res
