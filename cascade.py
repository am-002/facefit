import menpo.io as mio
from menpo.shape import PointCloud
import time

def shapeToLandmarkGroup(normal_shape):
    return LandmarkGroup(normal_shape, OrderedDict([('all', np.array([True]*68))]))

# TODO: Code duplication!!
def getBoundingBox(img):
    return img.landmarks['PTS'].lms.bounds()

def fitShapeToBox( shape, b):
    w = b[1][0] - b[0][0]
    h = b[1][1] - b[0][1]
    x = b[0][0]
    y = b[0][1]

    return PointCloud([ [p[0]*w+x,p[1]*h+y] for p in shape.points])


class Regressor:
    def train(data):
        pass
    def test(data):
        pass

class FeatureExtractor:
    def extract_features(data):
        pass

class Cascade(Regressor):
    def train(self, images_path, initial_shape):
        current_estimate = []

        print 'Starting at ', time.time()
        for t, r in enumerate(self.regressors):
            print 'Training regressor ', t
            print 'resizing images to 500x500!!'

            i = 0
            features = []
            targets = []
            for img in mio.import_images(images_path):
                img = img.resize((500,500))
                if (not img.has_landmarks):
                    continue
                if (len(current_estimate) <= i):
                    current_estimate.append(fitShapeToBox(initial_shape, getBoundingBox(img)))
                shape_indexed_features = r.extract_features((img, current_estimate[i]))
                delta = PointCloud(img.landmarks['PTS'].lms.points - current_estimate[i].points)

                features.append(shape_indexed_features)
                targets.append(delta)
                i += 1
            r.train(features, targets)

            for i in range(len(features)):
                current_estimate[i].points += r.test(features[i]).points

    def test(self, image, initial_shape):
        current_estimate = fitShapeToBox(initial_shape, getBoundingBox(image))

        for r in self.regressors:
            shape_indexed_features = r.extract_features((image, current_estimate))
            current_estimate.points += r.test(shape_indexed_features).points

        return current_estimate
