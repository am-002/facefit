from abc import abstractmethod


class FeatureExtractor(object):
    @abstractmethod
    def apply(self, image, shape):
        pass

class FeatureExtractorBuilder(object):
    @abstractmethod
    def build(self, images, shapes, targets, extra):
        pass

class Regressor(object):
    def apply(self, features, extra):
        pass

class RegressorBuilder(object):
    @abstractmethod
    def build(self, features, targets, extra):
        pass