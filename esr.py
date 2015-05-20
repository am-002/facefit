import hickle
import numpy as np
import menpo.io as mio
import menpo
from menpo.shape import PointCloud
from menpo.visualize import print_dynamic
from util import *
from fern_cascade import FernCascadeBuilder

import cv2

def get_gt_shape(image):
    return image.landmarks['PTS'].lms

def get_bounding_box(image, face_detector):
    #gray = cv2.cvtColor(image.pixels.astype(int), cv2.COLOR_RGB2GRAY)
    gray = image.pixels.astype(int)
    gray = gray.reshape((len(gray), len(gray[0]),))
    gray = np.array(gray, dtype=np.uint8)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    boxes = []
    maxi = 0
    for i, (x,y,w,h) in enumerate(faces):
	#TODO: We save the box in the menpo coordinate system.
        box = [y,x,h,w]
        boxes.append(box)
	if w > boxes[maxi][3]:
	     maxi = i;
	
    if len(faces) == 0:
	a, b = get_gt_shape(image).bounds()
	x, y = a
	w, h = b[0] - a[0], b[1] - a[1] 
        return np.array([x-0.05*w, y-0.05*h, 1.05*w, 1.05*h])
    #return np.array(boxes[0])
    #print boxes
    return boxes[maxi]


class ESRBuilder:
    def __init__(self, n_landmarks=68, n_stages=10, n_pixels=400, n_fern_features=5,
                 n_ferns=500, n_perturbations=20, kappa=0.3, beta=1000, stddev_perturb=0.04, basis_size=512):
        self.n_landmarks = n_landmarks
        self.n_stages = n_stages
        self.n_pixels = n_pixels
        self.n_ferns = n_ferns
        self.n_fern_features = n_fern_features
        self.n_perturbations = n_perturbations
        self.kappa = kappa
        self.beta = beta
        self.stddev_perturb = stddev_perturb
        self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.basis_size = basis_size

    @staticmethod
    def read_images(img_glob):
        # Read the training set into memory.
        images = []
        for img_orig in mio.import_images(img_glob, verbose=True, normalise=False):
            if not img_orig.has_landmarks:
                continue
            # Convert to greyscale and crop to landmarks.
            images.append(img_orig.as_greyscale(mode='average').crop_to_landmarks_proportion_inplace(0.5))
        return images

    # Fit a shape into a box. The shape has to be normalised and centered around
    # the origin (inside [-1, -1], [1, 1]).
    @staticmethod
    def fit_shape_to_box(normal_shape, box):
        #dim_x, dim_y = PointCloud(box).range()
	#print box
	(x, y, w, h) = box

        center_x = x + w*2.0/3.0
        center_y = y + h/2.0

        shape = normal_shape.points - normal_shape.centre()
        shape *= [0.9*h/2.0, 0.9*w/2.0]
        shape += [center_x, center_y]

        return PointCloud(shape)

    def from_file(self, file):
        return hickle.load(file, safe=False)

    def centered_mean_shape(self, target_shapes):
        mean_shape = menpo.shape.mean_pointcloud(target_shapes)
        return PointCloud(2 * (mean_shape.points - mean_shape.centre()) / mean_shape.range())

    def perturb_boxes(self, boxes, n_perturbations):
        widths = boxes[:, 1, 0] - boxes[:, 0, 0]
        heights = boxes[:, 1, 1] - boxes[:, 0, 1]

        ranges = np.dstack((widths, heights))[0]
        ranges = ranges.repeat(2, axis=0).reshape((len(ranges), 2, 2))
        ranges = ranges.repeat(n_perturbations, axis=0)

        normalized_box_offsets = np.random.normal(loc=0, scale=self.stddev_perturb, size=(len(boxes)*n_perturbations, 2, 2))
        return boxes.repeat(n_perturbations, axis=0) + normalized_box_offsets * ranges

    def perturb_shapes(self, shapes, n_perturbations):
        # TODO: Get rid of magic numbers.
        #dx = np.random.normal(loc=0, scale=0.03, size=(len(shapes)))
        dx = np.random.uniform(low=-0.1, high=0.1, size=(len(shapes)))
        
	# Less variance in horizontal direction?
        #dy = np.random.normal(loc=0, scale=0.01, size=(len(shapes)))
        dy = np.random.uniform(low=-0.1, high=0.1, size=(len(shapes)))

	# TODO: In menpo, pt (x, y) is stored as [y, x].
        normalized_offsets = np.dstack((dy, dx))[0]
	
	ret = []
        for i in xrange(len(shapes)):
            ret.append(PointCloud(shapes[i].points + shapes[i].range() * normalized_offsets[i]))
	return ret
	    

    def get_gt_shapes(self, images):
        return [img.landmarks['PTS'].lms for img in images]

    def get_bounding_boxes(self, images):
        ret = []
        for i, image in enumerate(images):
             print_dynamic("Detecting face {}/{}".format(i, len(images)))
             ret.append(get_bounding_box(image, self.face_detector))
        return ret

    def build(self, images):
        images = np.array(self.read_images(images))
        self.mean_shape = self.centered_mean_shape([img.landmarks['PTS'].lms for img in images])

        # Generate initial shapes with perturbations.
	boxes = np.array(self.get_bounding_boxes(images))
	print_dynamic('Generating initial shapes')
	shapes = np.array([self.fit_shape_to_box(self.mean_shape, box) for box in boxes])
	
	boxes = boxes.repeat(self.n_perturbations, axis=0) 
        images = images.repeat(self.n_perturbations, axis=0)
        shapes = shapes.repeat(self.n_perturbations, axis=0)
	
	shapes = self.perturb_shapes(shapes, self.n_perturbations)

        # Extract ground truth shapes from annotated images.
        gt_shapes = self.get_gt_shapes(images)

	assert(len(boxes) == len(images))
	assert(len(shapes) == len(images))
	assert(len(gt_shapes) == len(images))

	print('\nSize of augmented dataset: {} images.\n'.format(len(images)))

        fern_cascades = []
        for j in xrange(self.n_stages):
            fern_cascade_builder = FernCascadeBuilder(self.n_pixels, self.n_fern_features, self.n_ferns,
                                                      self.n_landmarks, self.mean_shape, self.kappa, self.beta, self.basis_size)
            fern_cascade = fern_cascade_builder.build(images, shapes, gt_shapes)
            # Update current estimates of shapes.
            #shapes = [fern_cascade.apply(image, shape, transform_to_mean_shape(shape, self.mean_shape).pseudoinverse())
            #          for image, shape in zip(images, shapes)]
            for i, (image, shape) in enumerate(zip(images, shapes)):
                offset = fern_cascade.apply(image, shape, transform_to_mean_shape(shape, self.mean_shape).pseudoinverse())
                #  'Got offset[{}] = '.format(i), offset.points
                # if i == 0:
                    # print 'Got offset: ', offset.points
                shapes[i].points += offset.points
            fern_cascades.append(fern_cascade)
            print("\nBuilt outer regressor {}\n".format(j))

        return ESR(self.n_landmarks, fern_cascades, self.mean_shape, self.face_detector)

class ESR:
    def __init__(self, n_landmarks, fern_cascades, mean_shape, face_detector):
        self.n_landmarks = n_landmarks
        self.fern_cascades = fern_cascades
        self.mean_shape = mean_shape
        self.face_detector = face_detector

    def fit(self, image, initial_shape):
        assert(initial_shape.n_points == self.n_landmarks)
        image = image.as_greyscale(mode='average')
        shape = ESRBuilder.fit_shape_to_box(initial_shape, get_bounding_box(image, self.face_detector))

        # print 'initial shape in fitter: ', shape.points

        for r in self.fern_cascades:
            mean_to_shape = transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
            # normalized_target = r.apply(image, shape, mean_to_shape)
            # shape.points += mean_to_shape.apply(normalized_target).points
            offset = r.apply(image, shape, mean_to_shape)
            # print 'Regressed offset: ', offset.points
            # print 'Regressed offset ', offset.points
            shape.points += offset.points
        return shape
