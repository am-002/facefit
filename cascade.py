import menpo.io as mio
from menpo.shape import PointCloud

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
    def extract_features(data, ):
        pass

class Cascade(Regressor):
    def train(self, images_path, initial_shape):
        current_estimate = []

        for t, r in enumerate(self.regressors):
            training_data = []
            print 'Training regressor ', t

            i = 0
            for img in mio.import_images(images_path):
                if (not img.has_landmarks):
                    continue
                if (len(current_estimate) <= i):
                    current_estimate.append(fitShapeToBox(initial_shape, getBoundingBox(img)))
                print 'Extracting features for image ', i
                shape_indexed_features = r.extract_features((img, current_estimate[i]))
                delta = PointCloud(img.landmarks['PTS'].lms.points - current_estimate[i].points)
                training_data.append((shape_indexed_features, delta))
                i += 1
            print 'Training data extracted'
            r.train(training_data)

            for i, (shape_indexed_features, delta) in enumerate(training_data):
                current_estimate[i].points += r.test(shape_indexed_features).points

    def test(self, image, initial_shape):
        current_estimate = fitShapeToBox(initial_shape, getBoundingBox(image))

        for r in self.regressors:
            shape_indexed_features = r.extract_features((image, current_estimate))
            current_estimate.points += r.test(shape_indexed_features).points

        return current_estimate
