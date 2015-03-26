from cascade import Cascade
from fern import Fern;
from feature_selector import FeatureSelector
from landmark_feature_selector import PixelDifferenceExtractor
import menpo.io as mio
import menpo
import numpy as np
from menpo.shape import PointCloud


def getNormalisedMeanShape(img_path):
        mean_shape = menpo.shape.mean_pointcloud([img.landmarks['PTS'].lms for img in mio.import_images(img_path)])
        b = mean_shape.bounds()
        w = b[1][0] - b[0][0]
        h = b[1][1] - b[0][1]
        x = b[0][0]
        y = b[0][1]
        normal_shape = PointCloud([ [(p[0] - x)/w, (p[1]-y)/h] for p in mean_shape.points])
        return normal_shape

class ExplicitShapeRegressor(Cascade):


    def __init__(self, nLandmarks, nRegressors, P, nFernFeatures, nFerns):
        self.nLandmarks = nLandmarks
        self.nRegressors = nRegressors
        self.nPixels = P
        self.nFerns = nFerns

        # Calculate mean shape from a subset of training data.
        self.mean_shape = getNormalisedMeanShape('../helen/subset_cropped/')
        self.regressors = [PrimaryRegressor(P, nFernFeatures, nFerns, nLandmarks, self.mean_shape) for i in range(nRegressors)]

class PrimaryRegressor:
    # landmarkFeatureSelector
    # primitive_regressors

    def __init__(self, P, F, nFerns, nLandmarks, mean_shape):
        self.nFerns = nFerns
        self.nFernFeatures = F
        self.nFeatures = P
        self.nLandmarks = nLandmarks
        # TODO: Pass argumetns for FeatureExtractorConstructor in a dictionary
        self.feature_extractor = PixelDifferenceExtractor(P, nLandmarks, mean_shape)
        self.primitive_regressors = [PrimitiveRegressor(P, F, nFerns, nLandmarks) for i in range(nFerns)]

    def extract_features(self, data):
        img = data[0]
        current_estimate = data[1]

        return self.feature_extractor.extract_features(img, current_estimate)

    def train(self, features, targets):
        #cov_pp = np.cov(features)
        aux = ([ [pixels[i] for pixels in features] for i in range(self.nFeatures)])
        cov_pp = np.cov(aux)

        print 'Training primitive regressors '
        for i, r in enumerate(self.primitive_regressors):
            print i
            r.train(features, targets, cov_pp)
            for j in range(len(features)):
                targets[j].points -= r.test(features[j]).points


    def test(self, shape_indexed_features):
        res = PointCloud([ [0,0] for i in range(self.nLandmarks)])
        for r in self.primitive_regressors:
            offset = r.test(shape_indexed_features)
            res.points += offset.points
        #res.points /= float(self.nFerns)
        return res

class PrimitiveRegressor:
    def __init__(self, nFeatures, nFernFeatures, nFerns, nLandmarks):
        self.nFerns = nFerns
        self.nFernFeatures = nFernFeatures
        self.nFeatures = nFeatures
        self.nLandmarks = nLandmarks
        self.fern = Fern(nFernFeatures, nLandmarks)
        self.fern_feature_selector = FeatureSelector(nFeatures, nFernFeatures)

    def train(self, features, targets, cov_pp):
        self.fern_feature_selector.train(features, targets, cov_pp)
        fern_training_data = []
        for i in range(len(features)):
            fern_features = self.fern_feature_selector.extract_features(features[i])
            fern_training_data.append((fern_features, targets[i]))
        self.fern.train(fern_training_data)

    def test(self, shapeIndexedFeatures):
        fern_features = self.fern_feature_selector.extract_features(shapeIndexedFeatures)

        return self.fern.test(fern_features)
