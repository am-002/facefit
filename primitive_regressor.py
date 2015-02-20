from feature_selector import FeatureSelector
from fern import Fern

class PrimitiveRegressor:
    # fern_feature_selector
    # fern

    def __init__(self, nFeatures, nFernFeatures, nFerns):
        self.fern = Fern(nFernFeatures)
        self.fern_feature_selector = FeatureSelector(nFeatures, nFernFeatures)

    def train(self, training_data):
        self.fern_feature_selector.train(training_data)
        fern_training_data = []
        for (shapeIndexedFeatures, offset) in training_data:
            fern_training_data.append((self.fern_feature_selector.selectFeatures(shapeIndexedFeatures), offset))
        self.fern.train(fern_training_data)

    def getOffset(self, shapeIndexedFeatures):
        return self.fern.getOffset(self.fern_feature_selector.selectFeatures(shapeIndexedFeatures))

