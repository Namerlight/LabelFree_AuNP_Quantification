import os
import numpy as np
import scipy.ndimage
from torchvision.io import read_image

from PIL import Image

def generate_label_annotations(file_name: str):
    """
    Loads a density map image for the Synthetic Cells dataset and notes locations of dots that indicate a cell.

    Args:
        file_name:

    Returns:

    """

    image = read_image(file_name)
    image = image.detach().cpu().numpy()

    annotations = []

    for channel in image:
        for r_n, row in enumerate(channel):
            for c_n, column in enumerate(row):
                if column == 255:
                    annotations.append([r_n, c_n])

    return annotations


image_name = ""
image_paths = os.path.join("..", "data", "synthetic-cells", "val")

image_paths = [os.path.join(image_paths, file) for file in os.listdir(image_paths)]
labels_path = []

for _, path in enumerate(image_paths):
    if path[-8:] != "cell.png":
        image_paths.pop(_)

print(image_paths)

for image_path in image_paths:

    img = np.array(Image.open(image_path), dtype=np.float32) / 255
    image = np.transpose(img, (2, 0, 1))

    label = np.array(Image.open(f"{image_path[:-8]}dots.png"))

    label = 100.0 * (label[:, :, 0] > 0)

    d_map = scipy.ndimage.gaussian_filter(label, sigma=(1, 1), order=0)
    d_map = d_map[np.newaxis, :, :, ]

    np.save(f"{image_path[:-8]}dmap", d_map)

print("All dots images in folder converted to density maps.")
