from menpo.shape import PointCloud
import menpo.io as mio
from primary_regressor import PrimaryRegressor
from menpo.transform import AlignmentSimilarity
import menpo
import sys

def getNormalisedMeanShape(img_path):
    mean_shape = menpo.shape.mean_pointcloud([img.landmarks['PTS'].lms for img in mio.import_images(img_path)])
    l = zip(*list(mean_shape.points))
    box = [ [min(l[0]), min(l[1])], [max(l[0]), max(l[1])] ]
    w = box[1][0] - box[0][0]
    h = box[1][1] - box[0][1]

    return PointCloud([((p[0]-box[0][0])/w, (p[1]-box[0][1])/h) for p in mean_shape.points])



def fit_shape_to_box(normal_shape, box):
    w = box[1][0] - box[0][0]
    h = box[1][1] - box[0][1]

    return PointCloud([ [p[0]*w + box[0][0], p[1]*h+box[0][1]] for p in normal_shape.points])



def get_bounding_box(lg):
    l = zip(*list(lg.lms.points))
    box = [ [min(l[0]), min(l[1])], [max(l[0]), max(l[1])] ]
    return box


class ExplicitShapeRegressor:
    def __init__(self, nLandmarks, nRegressors, P, nFernFeatures, nFerns):
        self.nLandmarks = nLandmarks
        self.nRegressors = nRegressors
        self.nPixels = P
        self.nFerns = nFerns

        # Calculate mean shape from a subset of training data.
        self.mean_shape = getNormalisedMeanShape('../helen/subset_cropped/')
        self.regressors = [PrimaryRegressor(P, nFernFeatures, nFerns, nLandmarks, self.mean_shape) for i in range(nRegressors)]

    def train(self, img_glob):
        init_shape = self.mean_shape
        n_samples = sum(1 for img in mio.import_images(img_glob) if img.has_landmarks)
        #shapes = [fit_shape_to_box(init_shape) for i in xrange(n_samples)]
        shapes = [fit_shape_to_box(init_shape, get_bounding_box(img.landmarks['PTS'])) for img in mio.import_images(img_glob)]


        for r in self.regressors:
            pixels = []
            targets = []
            sys.stdout.flush()
            for i, img in enumerate(mio.import_images(img_glob)):
                if not img.has_landmarks:
                    continue
                pixels.append(r.extract_features(img, shapes[i]))
                delta = PointCloud(img.landmarks['PTS'].lms.points - shapes[i].points)
                normalized_target = AlignmentSimilarity(shapes[i], self.mean_shape).apply(delta)
                targets.append(normalized_target)

            r.train(pixels, targets)

            for i in xrange(n_samples):
                normalized_offset = r.apply(shapes[i], pixels[i])
                offset = AlignmentSimilarity(shapes[i], self.mean_shape).pseudoinverse().apply(normalized_offset).points
                shapes[i].points += offset
                #shapes[i].points += r.apply(shapes[i], pixels[i]).points

    def fit(self, image, initial_shape):
        shape = fit_shape_to_box(initial_shape, get_bounding_box(image.landmarks['PTS']))

        for r in self.regressors:
            pixels = r.extract_features(image, shape)
            shape.points += AlignmentSimilarity(shape, self.mean_shape).pseudoinverse().apply(r.apply(shape, pixels)).points
            #shape.points += r.apply(shape, pixels).points

        return shape
