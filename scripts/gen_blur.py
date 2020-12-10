import numpy as np

import skimage


rgb = skimage.io.imread('../images/rgb.png')


blurred = skimage.filters.gaussian(rgb, sigma=10)

skimage.io.imsave('blurred.png', blurred)