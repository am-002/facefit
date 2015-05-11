import menpo.io as mio
import numpy as np
import esr

builder = esr.ESRBuilder(n_landmarks=68, n_stages=1, n_ferns=1, n_perturbations=1, beta=0)

images = builder.read_images("../../helen/two/*")

builder.build(images)

esr = builder.build(images, )

img = mio.import_image("../../helen/two/2500088309_1.jpg")

pc = esr.fit(img, builder.mean_shape)
print img.landmarks['PTS'].lms.points
print pc.points