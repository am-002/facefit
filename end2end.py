import menpo.io as mio
import numpy as np
import esr

import menpo.io as mio
from menpo.visualize import print_dynamic
from menpofit.fittingresult import compute_error
import numpy as np
from esr import ESRBuilder


builder = ESRBuilder()

trainset = "/Users/andrejm/Google Drive/Work/BEng project/helen/subset"
testset = "/Users/andrejm/Google Drive/Work/BEng project/helen/subset"



model = builder.build(trainset)

lfpw_test_images = []
for im in mio.import_images(testset, verbose=True, normalise=False):
    im.crop_to_landmarks_proportion_inplace(0.5)
    # im = im.as_grayscale(mode='average')
    lfpw_test_images.append(im)

initial_shape = model.mean_shape

initial_errors = []
final_errors = []
final_shapes = []
# initial_shapes = extract_shapes(lfpw_test_images)
initial_shapes = [ESRBuilder.fit_shape_to_box(img.landmarks['PTS'].lms, esr.get_bounding_box(img)) for img in lfpw_test_images]
for k, im in enumerate(lfpw_test_images):
    gt_shape = im.landmarks[None].lms
    final_shape = model.fit(im, model.mean_shape)

    final_shapes.append(final_shape)
    initial_shapes.append(initial_shape)

    initial_errors.append(compute_error(initial_shape, gt_shape))
    final_errors.append(compute_error(final_shape, gt_shape))

    print_dynamic('{}/{}'.format(k + 1, len(lfpw_test_images)))

print(np.mean(initial_errors))
print(np.mean(final_errors))

