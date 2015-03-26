import math
import random
import time
import numpy as np
from menpo.shape import PointCloud
from menpo.transform import ThinPlateSplines
from menpo.image import BooleanImage

class PixelDifferenceExtractor:
    def getClosestPoint(self, pc, pt):
        closest = 0
        closest_dist = 1234567
        for i,p in enumerate(pc.points):
            dist = math.hypot(p[0]-pt[0], p[1]-pt[1])
            if (dist < closest_dist):
                closest_dist = math.hypot(p[0]-pt[0], p[1]-pt[1])
                closest = i
        return closest

    def getIntensity(self, img, x, y):
        w = img.shape[0]
        h = img.shape[1]
        if (x < 0 or y < 0 or x >= w or y >= h):
            return 0
        # Assuming it's greyscale.
        return img.pixels[int(x)][int(y)][0]

    def getBoundingBox(self, img):
        return img.landmarks['PTS'].lms.bounds()

    def getMaskAroundBoundingBox(self, image):
        mask = np.ndarray(image.shape, dtype=bool)
        b = self.getBoundingBox(image)
        w = b[1][0] - b[0][0]
        h = b[1][1] - b[0][1]
        x = b[0][0]
        y = b[0][1]

        for i in range(int(max([0, x-w/4])), int(min([x+1.25*w, image.shape[0]]))):
            for j in range(int(max([0, y-h/4])), int(min([y+1.25*h, image.shape[1]]))):
                mask[i][j] = True
        return BooleanImage(mask)

    def fitShapeToBox(self, shape, b):
        w = b[1][0] - b[0][0]
        h = b[1][1] - b[0][1]
        x = b[0][0]
        y = b[0][1]

        return PointCloud([ [p[0]*w+x,p[1]*h+y] for p in shape.points])

    def __init__(self, nPixels, nLandmarks, mean_shape):
        self.nFeatures = nPixels;
        self.features = []
        self.mean_shape = mean_shape

        for i in range(nPixels):
            x = random.uniform(-0.25, 1.25)
            y = random.uniform(-0.25, 1.25)
            lmark = self.getClosestPoint(mean_shape, [x, y])
            dx = x - mean_shape.points[lmark][0]
            dy = y - mean_shape.points[lmark][1]
            self.features.append( (lmark, [dx, dy]) )


    def extract_features(self, image, current_estimate):
        ret = []


        now = time.time()
        # Wrap the image so that the current estimate aligns with mean shape,
        # then extract features.
        mean_shape_pc = self.fitShapeToBox(self.mean_shape, self.getBoundingBox(image))
        image_to_mean = ThinPlateSplines(mean_shape_pc, image.landmarks['PTS'].lms)

        # Warping the whole image so that the current estimate aligns with the mean_shape is too slow.
        # mask = image.as_masked().mask

        # Only warp the region around the bounding box.
        #mask = self.getMaskAroundBoundingBox(image)
        #warpped_to_mean = image.warp_to_mask(mask, image_to_mean)

        # No warping at all!
        warpped_to_mean = image



        now = time.time()

        b = self.getBoundingBox(image)
        w = b[1][0] - b[0][0]
        h = b[1][1] - b[0][1]

        ret = []
        for (lmark, offset) in self.features:

            x = current_estimate.points[lmark][0] + offset[0]*w
            y = current_estimate.points[lmark][1] + offset[1]*h
            ret.append(self.getIntensity(warpped_to_mean, x, y))


        #for (lmark1, offset1) in self.features:
        #    x1 = current_estimate.points[lmark1][0] + offset1[0]*warpped_to_mean.shape[0]
        #    y1 = current_estimate.points[lmark1][1] + offset1[1]*warpped_to_mean.shape[1]

            # Get intensity of pixel p at (x1, y1).
        #    p = self.getIntensity(warpped_to_mean, x1, y1)
        #    for (lmark2, offset2) in self.features:
        #        x2 = current_estimate.points[lmark2][0] + offset2[0]*warpped_to_mean.shape[0]
        #        y2 = current_estimate.points[lmark2][1] + offset2[1]*warpped_to_mean.shape[1]

                # Get intensity of pixel q at (x2, y2).
        #        q = self.getIntensity(warpped_to_mean, x2, y2)
                # Append the pixel difference p-q to the result.
        #        ret.append(p-q)
        return ret
