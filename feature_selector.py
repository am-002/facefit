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

    def average(self, x):
        return float(sum(x)) / len(x)

    def getPearson(self, tuples):
        x = []
        y = []
        for t in tuples:
            x.append(t[0])
            y.append(t[1])

         #return numpy.corrcoef(x, y)[0, 1]

        n = len(x)
        assert n > 0
        avg_x = self.average(x)
        avg_y = self.average(y)
        diffprod = 0
        xdiff2 = 0
        ydiff2 = 0
        for idx in range(n):
            xdiff = x[idx] - avg_x
            ydiff = y[idx] - avg_y
            diffprod += xdiff * ydiff
            xdiff2 += xdiff * xdiff
            ydiff2 += ydiff * ydiff

        print xdiff2
        print ydiff2
        return diffprod / math.sqrt(xdiff2 * ydiff2)


    def getRandomDirection(self):
        return [random.random()*10, random.random()*10]

    # Project a onto b and return length of the projection.
    def project(self, a, b):
        return (a[0]*b[0] + a[1]*b[1]) / math.sqrt((a[0])**2 + (a[1])**2)


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
                for origFeature in range(len(originalFeatures)):
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
        res = [0]*self.F
        for i in range(self.F):
            res[i] = original[self.features[i]]
        return res
