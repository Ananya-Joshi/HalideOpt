import numpy as np

import skimage


noise = np.random.uniform(0, 1, size=(512, 512, 3))

skimage.io.imsave('../images/noise.png', noise)