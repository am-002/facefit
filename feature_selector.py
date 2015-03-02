# Correlation-based feature selection.

import numpy as np
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

        # TODO
        if (xdiff2 == 0 or ydiff2 == 0):
            return 0
        return diffprod / math.sqrt(xdiff2 * ydiff2)

    # TODO: Random projection from unit gaussian?
    def getRandomDirection(self):
        return np.array([random.uniform(-1, 1) for i in range(2*68)])

    def magnitude(self, x):
        return np.sqrt(np.dot(x, x))

    # Project a onto b and return length of the projection.
    def project(self, a, b):
        return np.dot(a, b) / (self.magnitude(a) * self.magnitude(b));
        #return (a[0]*b[0] + a[1]*b[1]) / math.sqrt((a[0])**2 + (a[1])**2)


    def train(self, arr):
        for i in range(self.F):
            # Project the offset in random direction and measure the scalar length
            # of the projection. Find which feature has highest imapct on the length.

            d = self.getRandomDirection()
            featureVsLength = [None]*self.P2
            for j in range(self.P2):
                featureVsLength[j] = []

            for (originalFeatures, offset) in arr:
                l = self.project(offset.as_vector(), d)
                for origFeature in range(len(originalFeatures)):
                    f = originalFeatures[origFeature]
                    featureVsLength[origFeature].append([f, l])

            maxim = -123456.0
            res = 0.0
            for origFeature in range(self.P2):
                #print featureVsLength[origFeature]
                corr = self.getPearson(featureVsLength[origFeature])
                #print 'Corr is ', corr
                if corr > maxim:
                    res = origFeature
                    maxim = corr
            self.features[i] = res
            #print 'Picking feature ', res
            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).
        # Here the features array is populated

    def extract_features(self, original):
        res = [0]*self.F
        for i in range(self.F):
            res[i] = original[self.features[i]]
        return res
