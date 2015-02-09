import random

class LandmarkFeatureSelector:
    def __init__(self, nPixels):
        self.nPixels = nPixels;
        self.pixels = []
        for i in range(nPixels):
            # TODO: fixed region?
            dx = random.randrange(-200, 200)
            dy = random.randrange(-200, 200)
            self.pixels.append( (dx, dy) )

    def outOfBounds(self, image, p):
        (x, y) = p
        return x < 0 or y < 0 or x >= len(image) or y >= len(image[0])

    def getPixelDifferences(self, image, p1, p2):
        if self.outOfBounds(image, p1):
            i1 = 0
        else:
            (x, y) = p1
            i1 = image[x][y]

        if self.outOfBounds(image, p2):
            i2 = 0
        else:
            (x, y) = p2
            i2 = image[x][y]
        return i1-i2

    def getLandmarkFeatures(self, image, landmark):
        res = []
        lx = landmark[0]
        ly = landmark[1]
        for (dx, dy) in self.pixels:
            p1 = (lx+dx, ly+dy)
            for (dx2, dy2) in self.pixels:
                p2 = (lx+dx2, ly+dy2)
            res.append(self.getPixelDifferences(image, p1, p2))

        return res

