import os
import sys
import cv2 as cv
import numpy as np
import scipy.ndimage
from PIL import Image
from crop_cell_npn_images import *

np.set_printoptions(threshold=sys.maxsize)

def find_missing_files():

    dataset_directory = os.path.join("..", "data", "asmm")
    image_directory = os.path.join(dataset_directory, "cell_imgs_npns")

    og_imgs = [item[-8:] for item in os.listdir(os.path.join(image_directory, "normal"))]
    dmapped_imgs = [item[-8:] for item in os.listdir(os.path.join(image_directory, "normal_dmap"))]

    missing_dmapped = list(set(dmapped_imgs)-set(og_imgs))

    missing_ogs = list(set(og_imgs)-set(dmapped_imgs))

    print(sorted(missing_ogs))
    print(sorted(missing_dmapped))


def convert_to_dmap():
    dataset_directory = os.path.join("..", "data", "asmm")
    images_directory = os.path.join(dataset_directory, "cell_imgs_npns", "normal_dmap")

    image_paths = [os.path.join(images_directory, img_path) for img_path in os.listdir(images_directory)]

    for image_name in os.listdir(images_directory):
        image_path = os.path.join(images_directory, image_name)

        if image_name[-3:] == "png":
            print(image_name.split(".")[0][:-3])
            image = Image.open(image_path)
            image = image.resize((256, 256))
            image = image.convert('RGB')
            # image.save(image_path)

            if image_name.split(".")[0][-4:] == "L":
                opencvImage = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
                thresholded_image = threshold_image_fixed(opencvImage)
                cv.imshow("Image", thresholded_image)
                k = cv.waitKey(0)
                generate_density_map(image_path, thresholded_image)

    print(len(image_paths), "images converted to Density Map")


if __name__ == "__main__":
    convert_to_dmap()
