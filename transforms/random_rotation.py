""" Random Rotation """

import numpy as np
from torchvision.transforms import functional as TF


class RandomRotation(object):
    """Implementation of random rotation.

    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.

    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.

    Attributes:
        p:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.

    """

    def __init__(self, p: float = 0.5, angle: int = 90):
        self.p = p
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.

        Args:
            sample:
                PIL image which will be rotated.

        Returns:
            Rotated image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.p:
            sample = TF.rotate(sample, self.angle)
        return sample
