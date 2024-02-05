import os
import sys
import cv2 as cv
import numpy as np
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def open_image(path: str):
    return Image.open(path)


def resize_image(img, size: int = 256):
    img = img.resize((size, size))
    print("Image Resized")
    return img


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def threshold_image_fixed(img):
    ret, thresh4 = cv.threshold(img, 254.99, 255, cv.THRESH_TOZERO)
    images = [img, thresh4]
    return images[1]


def generate_density_map(image_path, thresholded_img):

    thresholded_image = np.array(thresholded_img)

    label = 100.0 * (thresholded_image[:, :, 0] > 0)

    d_map_2d = scipy.ndimage.gaussian_filter(label, sigma=(4, 4), order=0)
    d_map = d_map_2d[np.newaxis, :, :, ]

    plt.imshow(d_map_2d, cmap="hot")
    plt.show()

    # np.save(f"{image_path[:-4]}dmap", d_map)


if __name__ == "__main__":

    resize = False
    folder = True

    image_paths = os.path.join("..", "data", "asmm", "cell_imgs_npns", "normal")
    label_paths = os.path.join("..", "data", "asmm", "cell_imgs_npns", "normal_annot")

    if folder:
        for image_name in os.listdir(image_paths):
            image_path = os.path.join(image_paths, image_name)

            print(image_path)
            if resize:
                image = open_image(image_path)
                image = resize_image(image, size=256)
                image = image.convert('RGB')
                image.save(image_path)

        for label_name in os.listdir(label_paths):
            label_path = os.path.join(label_paths, label_name)

            label = open_image(label_path)
            print(label_path)

            opencvImage = cv.cvtColor(np.array(label), cv.COLOR_RGB2BGR)
            thresholded_image = threshold_image_fixed(opencvImage)
            for i, r in enumerate(thresholded_image):
                for j, p in enumerate(r):
                    if np.sum(p) != 0 and np.sum(p) < 765:
                        thresholded_image[i][j] = np.asarray([0, 0, 0])
            # cv.imshow("Image", thresholded_image)
            # k = cv.waitKey(0)
            generate_density_map(label_path, thresholded_image)

    else:
        image = open_image(os.listdir(image_paths)[0])
        image = resize_image(image, size=256)
        image.show()
        opencvImage = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        thresholded_image = threshold_image_fixed(opencvImage)
        cv.imshow("Image", thresholded_image)
        k = cv.waitKey(0)

