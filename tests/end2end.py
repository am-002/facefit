import menpo.io as mio
import numpy as np
import esr

builder = esr.ESRBuilder(n_landmarks=68, n_stages=10, n_ferns=500, n_perturbations=1, beta=1000, stddev_perturb=0.000001)

images = builder.read_images("../../helen/subset/*")

esr = builder.build(images)

img = mio.import_image("../../helen/subset/10405146_1.jpg")

pc = esr.fit(img, builder.mean_shape)
print img.landmarks['PTS'].lms.points
print pc.points