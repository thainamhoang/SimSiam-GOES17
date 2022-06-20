""" GOES17 DATASET """

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from typing import List, Tuple

MAIN_PATH = "./images"


class GOES17Dataset(Dataset):
    """Set up Dataset for GOES17 preprocessed images.

        Iterates through list of GOES17 images and applies transformations on the given image.

        Returns a tuple of 2 post-transformed images for twin network.

        Attributes:
            img_path:
                List of image paths
            transform:
                Transformation for each image

    """

    def __init__(self,
                 img_path: List,
                 transform: T.Compose = None) -> None:
        super(GOES17Dataset, self).__init__()
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.img_path[idx]
        img = Image.open(f"{MAIN_PATH}/{img_name}").resize((224, 224))

        # Apply transformation
        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2

    def __len__(self) -> int:
        return len(self.img_path)