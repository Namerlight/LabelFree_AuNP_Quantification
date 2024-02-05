import os
import torch
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class CellsNPNDataset(Dataset):
    """
    Dataset class for custom NPNs in Cells dataset.
    """

    def __init__(self, dataset_dir: str = None, transform=None, target_transform=None) -> None:

        images_names_list = [file.split(os.path.sep)[-1].split(".")[0].split("dmap")[0] for file in os.listdir(dataset_dir)]

        # Getting each unique image rather than both images and ground truth together
        self.images = [os.path.join(dataset_dir, name) for name in images_names_list]
        self.transform = transform
        self.target_transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        image = read_image(str(self.images[item]) + ".png")
        dmap = np.load(str(self.images[item]) + "dmap.npy")

        dmap = torch.from_numpy(dmap)

        # print(image.shape, type(image), dmap.shape, type(dmap))

        image = self.transform(image)
        dmap = self.transform(dmap)

        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        dmap = resize(dmap)

        rotate = transforms.RandomRotation((5, 85))

        if random.random() > 0.9:
            image = TF.hflip(image)
            dmap = TF.hflip(dmap)

        if random.random() > 0.9:
            image = TF.vflip(image)
            dmap = TF.vflip(dmap)

        if random.random() > 0.9:
            image = rotate(image)
            dmap = rotate(dmap)

        toTens = transforms.ToTensor()

        image = toTens(image)
        dmap = toTens(dmap)

        return image, dmap
