from cascade import Cascade
from fern import Fern;
from feature_selector import FeatureSelector
from landmark_feature_selector import PixelDifferenceExtractor
import menpo.io as mio
import menpo
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
        self.nFeatures = P*P
        self.nLandmarks = nLandmarks
        # TODO: Pass argumetns for FeatureExtractorConstructor in a dictionary
        self.feature_extractor = PixelDifferenceExtractor(P, nLandmarks, mean_shape)
        self.primitive_regressors = [PrimitiveRegressor(P*P, F, nFerns, nLandmarks) for i in range(nFerns)]

    def extract_features(self, data):
        img = data[0]
        current_estimate = data[1]

        return self.feature_extractor.extract_features(img, current_estimate)

    def train(self, training_data):
        for i, r in enumerate(self.primitive_regressors):
            print 'Training primitive regressor ', i
            r.train(training_data)

    def test(self, shape_indexed_features):
        res = PointCloud([ [0,0] for i in range(self.nLandmarks)])
        for r in self.primitive_regressors:
            offset = r.test(shape_indexed_features)
            print 'Offset ', offset
            res.points += offset.points
        res.points /= float(self.nFerns)
        return res

class PrimitiveRegressor:
    def __init__(self, nFeatures, nFernFeatures, nFerns, nLandmarks):
        self.nFerns = nFerns
        self.nFernFeatures = nFernFeatures
        self.nFeatures = nFeatures
        self.nLandmarks = nLandmarks
        self.fern = Fern(nFernFeatures, nLandmarks)
        self.fern_feature_selector = FeatureSelector(nFeatures, nFernFeatures)

    def train(self, training_data):
        self.fern_feature_selector.train(training_data)
        fern_training_data = []
        for (shapeIndexedFeatures, offset) in training_data:
            fern_training_data.append((self.fern_feature_selector.extract_features(shapeIndexedFeatures), offset))
        self.fern.train(fern_training_data)

    def test(self, shapeIndexedFeatures):
        fern_features = self.fern_feature_selector.extract_features(shapeIndexedFeatures)

        return self.fern.test(fern_features)
