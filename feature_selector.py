# Correlation-based feature selection.

import numpy as np
import random
import math

class FeatureSelector:
    def __init__(self, nPixels, F):
        self.nPixels = nPixels
        self.F = F
        self.features = [(0,0)]*F
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

    def cov(self, x, y):
        mean_x = self.average(x)
        mean_y = self.average(y)
        res = 0.0
        for i in range(len(x)):
            res += (x[i]-mean_x) * (y[i]-mean_y)
        return res / float(len(x))


    def train(self, features, targets, cov_pp):
        for f in range(self.F):
            # Project the offset in random direction and measure the scalar length
            # of the projection. Find which feature has highest imapct on the length.

            d = self.getRandomDirection()

            l = []
            for target in targets:
                l.append(self.project(target.as_vector(), d))

            cov_l_p = [0]*self.nPixels
            for pixel in range(len(features[0])):
                p = []
                for i in range(len(features)):
                    p.append(features[i][pixel])
                #cov_l_p[pixel] = self.cov(l, p)
                cov_l_p[pixel] = self.cov(l, p)

            var_l = self.cov(l, l)

            maxcorr = -12345
            res = (0, 0)
            for i in range(self.nPixels):
                for j in range(self.nPixels):
                    # We want to get corr(l, pixel_i - pixel_j).
                    denom = var_l * (cov_pp[i][i]+cov_pp[j][j]-2*cov_pp[i][j])
                    if (denom <= 0):
                        # TODO: What to do in this case?
                        corr = 0
                    else:
                        corr = (cov_l_p[i] - cov_l_p[j]) / np.sqrt(denom)
                    if corr > maxcorr:
                        maxcorr = corr
                        res = (i, j)
            self.features[f] = res

            # TODO: How to make sure that this feature will not be chosen again? (If
            # it is correlated with another random projection length).
            # Here the features array is populated

    def extract_features(self, original):
        res = [0]*self.F
        for i in range(self.F):
            res[i] = original[self.features[i][0]] - original[self.features[i][1]]
        return res
