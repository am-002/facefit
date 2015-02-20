import copy
from landmark_feature_selector import LandmarkFeatureSelector;
from fern import Fern;
from feature_selector import FeatureSelector;
from primary_regressor import PrimaryRegressor;

import menpo

class ExplicitShapeRegressor:
    def __init__(self, T, P, F, nFerns, nLandmarks):
        self.T = T
        self.P = P
        self.F = F
        self.nFerns = nFerns
        self.primary_regressors = []*T
        self.nLandmarks = nLandmarks
        for i in range(T):
            self.primary_regressors.append(PrimaryRegressor(P, F, nFerns))

        # Calculate mean shape from a subset of training data.
        self.mean_shape = menpo.shape.mean_pointcloud([ img.landmarks['PTS'].lms for img in menpo.io.import_images('../helen/subset_cropped/*') ])

    def train(self, imgs_path):
        MAX_IMGS = 500
        current_shape = [None]*MAX_IMGS
        for i in range(MAX_IMGS):
            current_shape[i] = copy.deepcopy(self.mean_shape.points)

        for t in range(self.T):
            print 'RUNNING PRIMARY REGRESSOR NUMBER {0}'.format(t)
            training_data = []
            testing_data = []

            # Extract features and correct offsets to construct training_data.
            cur_img = 0
            for img in menpo.io.import_images(imgs_path):
                landmarks = img.landmarks['PTS'].lms.points
                # Every picture is converted into greyscale!
                grey_img = img.as_greyscale().pixels
                for l in range(len(landmarks)):
                    lmark = landmarks[l]
                    lmark_features = (self.primary_regressors[t].getLandmarkFeatures(grey_img, lmark))
                    offset = [lmark[0]-current_shape[cur_img][l][0], lmark[1]-current_shape[cur_img][l][1]]
                    training_data.append((lmark_features, offset))
                    print 'Finished landmark l={0}'.format(l)
                print 'Finished img cur_img={0}'.format(cur_img)
                cur_img += 1

            print 'About to start training regressor {0}'.format(t)
            # Train the t-th first level regressor on the training data.
            self.primary_regressors[t].train(training_data)


            # Run the t-th first level regressor on the training data, updating the shape estimates.
            cur_img = 0
            for img in menpo.io.import_images(imgs_path):
                print 'Updating shape of img {0}'.format(cur_img)
                grey_img = img.as_greyscale().pixels
                for l in range(self.nLandmarks):
                    lmark_features = self.primary_regressors[t].getLandmarkFeatures(grey_img, current_shape[cur_img][l])

                    print 'Testing primary regressor {0} with landmark {1}'.format(t, l)
                    res = self.primary_regressors[t].getOffset(lmark_features)

                    # Update the current estimate
                    current_shape[cur_img][l][0] += res[0]
                    current_shape[cur_img][l][1] += res[1]
                cur_img += 1

    def getShape(self, image):
        # run face detector
        current_shape = copy.deepcopy(self.mean_shape.points)
        grey_img = image.as_greyscale().pixels
        for t in range(self.T):
            for l in range(self.nLandmarks):
                lmark_features = self.primary_regressors[t].getLandmarkFeatures(grey_img, current_shape[l])
                res = self.primary_regressors[t].getOffset(lmark_features)

                # Update the current estimate
                current_shape[l][0] += res[0]
                current_shape[l][1] += res[1]

        return current_shape

