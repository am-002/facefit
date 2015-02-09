from landmark_feature_selector import LandmarkFeatureSelector;
from fern import Fern;
from feature_selector import FeatureSelector;

import menpo


class ExplicitShapeRegressor:
    def __init__(self, T, P, F, nFerns):
        self.T = T
        self.P = P
        self.F = F
        self.ferns = []*nFerns
        for i in range(nFerns):
            self.ferns.append(Fern(F))
        self.feature_selectors = []
        for j in range(nFerns):
            self.feature_selectors.append(FeatureSelector(P*P, F))

        self.landmarkFeatureSelector = [None]*T
        for k in range(T):
            self.landmarkFeatureSelector[k] = LandmarkFeatureSelector(P)

        # Calculate mean shape from a subset of training data.
        self.mean_shape = menpo.shape.mean_pointcloud([ img.landmarks['PTS'].lms for img in menpo.io.import_images('../helen/subset_cropped/*') ])

    def train(self, imgs_path):
        current_estimate = [None]*500
        for t in range(self.T):
            training_data = []
            which = 0
            for img in menpo.io.import_images(imgs_path):
                if t == 0:
                    current_estimate[which] = self.mean_shape.points
                for lmark in img.landmarks['PTS'].lms.points:
                    lmark_features = (self.landmarkFeatureSelector[t].getLandmarkFeatures(img.pixels, lmark))
                    print lmark
                    training_data.append((lmark_features,
                        [lmark[0]-current_estimate[which][0], lmark[1]-current_estimate[which][1]]))
                which += 1
            self.feature_selectors[t].train(training_data)
            for fern in self.ferns:
                fern.train(training_data)


    def test(self, image):
        # run face detector
        current_estimate = self.mean_shape
        for t in range(self.T):
            for i in range(self.nLandmarks):
                lmark = []
                lmark[0] = img.landmarks['PTS'].lms.points[2*i]
                lmark[1] = img.landmarks['PTS'].lms.points[2*i+1]
                lmark_features = (self.landmarkFeatureSelector[t].getLandmarkFeatures(img.pixels, lmark))

                sum_offsets = [0, 0]
                ff = 0.0
                for fern in self.ferns:
                    fern_features = self.feature_selector[ff]
                    ff += 1.0
                    offset = fern.getOffset(fern_features)
                    sum_offsets[0] += offset[0]
                    sum_offsets[1] += offset[1]
                sum_offsets[0] /= ff
                sum_offsets[1] /= ff

                current_estimate[2*i] += sum_offsets[0]
                current_estimate[2*i+1] += sum_offsets[1]

# e = ExplicitShapeRegressor(T=10, P=100, F=5, nFerns=100)

