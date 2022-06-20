""" TwoCropsTransform """

import torchvision.transforms as T
from typing import Any, List


class TwoCropsTransform(object):
    """TwoCropsTransform

    Take two random crops of one image as the query and key.

    Attributes:
        base_transform:
            Base transformation for the twin network

    """
    def __int__(self, base_transform: T.Compose):
        self.base_transform = base_transform

    def __call__(self, x: Any) -> List[Any]:
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
