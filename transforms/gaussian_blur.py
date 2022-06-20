""" Gaussian Blur """

import numpy as np
from PIL import ImageFilter
from typing import List


class GaussianBlur(object):
    """Implementation of random Gaussian blur.

    Utilizes the built-in ImageFilter method from PIL to apply a Gaussian
    blur to the input image with a certain probability. The blur is further
    randomized as the kernel size is chosen randomly around a mean specified
    by the user.

    Attributes:
        kernel_size:
            Mean kernel size for the Gaussian blur.
        p:
            Probability with which the blur is applied.
        scale:
            Fraction of the kernel size which is used for upper and lower
            limits of the randomized kernel size.

    """

    def __init__(self, sigma: List[float] = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, sample):
        """Blurs the image with a given probability.

        Args:
            sample:
                PIL image to which blur will be applied.

        Returns:
            Blurred image or original image.

        """
        sigma = np.random.uniform(*self.sigma)
        sample = sample.filter(ImageFilter.GaussianBlur(radius=sigma))
        return sample
