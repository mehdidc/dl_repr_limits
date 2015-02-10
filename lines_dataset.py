import numpy as np
import copy
from theano import config
import theano.tensor as T

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.utils.rng import make_np_rng
from skimage.draw import line

def gen(number, shape, rng):
    images = np.zeros( tuple([number] + list(shape)) )
    x, y = shape[0]/2, shape[1]/2
    max_r = 0.5*np.sqrt(x**2 + y**2)
    hidden = []
    for i in xrange(number):
        theta = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(0, max_r)
        rr, cc = line(x, y, x + int(r * np.cos(theta)), y + int(r * np.sin(theta)))
        rr = rr[rr < shape[0]]
        cc = cc[cc < shape[1]]
        images[i, rr, cc] = 1
        hidden.append( (theta, r) )
    labels  = np.array(hidden)
    return images.astype(np.float32), labels.astype(np.float32)

class Lines(DenseDesignMatrix):

    def __init__(self, shape=20, nb=100, switch_images_by_labels=False):
        self.shape = shape
        images, labels = gen(nb, (self.shape, self.shape), rng=np.random)
        images = images.reshape( (images.shape[0], np.prod(images.shape[1:])) )

        if switch_images_by_labels is True:
            images, labels = labels, images
        super(Lines, self).__init__(X=images, y=labels, view_converter=DefaultViewConverter( (shape, shape, 1)  ) )
