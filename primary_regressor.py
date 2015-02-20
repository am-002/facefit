from landmark_feature_selector import LandmarkFeatureSelector
from primitive_regressor import PrimitiveRegressor

class PrimaryRegressor:
    # landmarkFeatureSelector
    # primitive_regressors

    def __init__(self, P, F, nFerns):
        self.landmark_feature_selector = LandmarkFeatureSelector(P)
        self.primitive_regressors = []*nFerns
        for i in range(nFerns):
            self.primitive_regressors.append(PrimitiveRegressor(nFeatures=P*P, nFernFeatures=F, nFerns=nFerns))

    def getLandmarkFeatures(self, image, landmark):
        return self.landmark_feature_selector.getLandmarkFeatures(image, landmark)

    # Training data is an array of (shapeIndexedFeatures, offset) tuples.
    def train(self, training_data):
        for regressor in self.primitive_regressors:
            regressor.train(training_data)

    def getOffset(self, shapeIndexedFeatures):
        res = [0, 0]
        for regressor in self.primitive_regressors:
            offset = regressor.getOffset(shapeIndexedFeatures)
            res[0] += offset[0]
            res[1] += offset[1]
        n = len(shapeIndexedFeatures)
        return [res[0]/n, res[1]/n]
