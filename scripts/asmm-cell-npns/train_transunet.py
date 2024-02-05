import sklearn.metrics as metrics
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
import numpy as np
from torchsummary import summary
import os
import model_archs
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", module="matplotlib\..*")


def train(dataset: str, hyperparameters: dict, train_path: str, val_path: str, output_path: str):

    cells_dataset_train = datasets.CellsNPNDataset(dataset_dir=os.path.join("..", "..", train_path),
                                                         transform=datasets.image_transforms(dataset))
    cells_dataset_val = datasets.CellsNPNDataset(dataset_dir=os.path.join("..", "..", val_path),
                                                       transform=datasets.image_transforms(dataset))