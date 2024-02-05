import os
import scipy
import itertools
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def open_image(path: str, show_images: bool = False):
    img = Image.open(path)
    if show_images:
        img.show("Original Image input")
    return img


def crop_image(img, size: int) -> []:
    height, width = img.size

    print("Cropping Images")

    if height <= size or width <= size:
        return [img], [1, 1]

    num_row_slices, num_col_slices = height // size - 1, width // size - 1
    imgs = []

    for h in range(num_row_slices):
        for w in range(num_col_slices):
            imgs.append(img.crop((size * h, size * w, size * (h + 1), size * (w + 1))))

    return imgs, [num_row_slices, num_col_slices]


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    if isinstance(imgs[0], np.ndarray):
        imgs = [Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)) for img in imgs]

    black_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    w, h = imgs[0].size

    grid = Image.new('RGB', size=(rows * w, cols * h))

    cell_list = []

    for img, (row, col) in zip(imgs, itertools.product(range(rows), range(cols))):
        if (row, col) in [(0, 4), (1,3), (1,4), (2, 3),(2, 4),(3, 2),(3, 3),(3, 4),(3, 5),(4, 5), (4, 3), (4, 4), (5, 3), (5, 5)]:
            grid.paste(img, box=(row * w, col * h))
            cell_list.append(img)
        else:
            grid.paste(black_img, box=(row * w, col * h))


    grid.show("Smaller cropped images stitched back together in order.")

    for img, (row, col) in zip(imgs, itertools.product(range(rows), range(cols))):
        opencvImage = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        # cv.imshow(str(row)+" "+str(col), opencvImage)
        # k = cv.waitKey(0)

    return grid, cell_list


def threshold_image_fixed(img, threshold_value: int = 100):
    ret, thresh4 = cv.threshold(img, threshold_value, 255, cv.THRESH_TOZERO)
    images = [img, thresh4]
    return images[1]


def threshold_image_for_dmap(img):
    ret, thresh4 = cv.threshold(img, 0, 254, cv.THRESH_TOZERO)
    images = [img, thresh4]
    return images[1]


def generate_density_map(image_path, thresholded_img):

    thresholded_image = np.array(thresholded_img)

    label = 100.0 * (thresholded_image[:, :, 0] > 0)

    d_map = scipy.ndimage.gaussian_filter(label, sigma=(1, 1), order=0)
    d_map = d_map[np.newaxis, :, :, ]

    np.save(f"{image_path[:-5]}dmap", d_map)


def process_image(
        image_idx: int,
        image_path: str,
        crop_and_split: bool = True,
        threshold: bool = True,
        show_images: bool = True,
        save_images: bool = False,
        save_directory: str = None,
        cell_only: bool = False
):

    image = open_image(path=image_path, show_images=False)

    if crop_and_split:
        cropped_images, dim = crop_image(image, 256)
    else:
        cropped_images, dim = [image], (1, 1)

    if show_images:
        grid, cropped_images = image_grid(cropped_images, rows=dim[0], cols=dim[1])
        grid.show("Smaller cropped images stitched back together in order.")

    thresholded_images = []

    if threshold:

        for image in cropped_images:
            opencvImage = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
            thresholded_image = threshold_image_fixed(opencvImage, threshold_value=75)
            thresholded_images.append(thresholded_image)

        if show_images:
            for idx, thresholded_image in enumerate(thresholded_images):
                cv.imshow(str(idx), thresholded_image)
                k = cv.waitKey(0)

            grid = image_grid(thresholded_images, rows=dim[0], cols=dim[1])
            grid.show("Thresholded images stitched back together in order.")

    if cell_only:
        for idx, cropped_image in enumerate(cropped_images):
            cropped_image.save(os.path.join(save_directory, f"NPNs_{image_idx}_{idx}.png"))

    # if save_images:
    #     if threshold:
    #         for idx, thresholded_image in enumerate(thresholded_images):
    #             cv.imwrite(os.path.join(save_directory, f"NPNs_thresholded_{image_idx}_{idx}.png"), thresholded_image)
    #     else:
    #         for idx, cropped_image in enumerate(cropped_images):
    #             cropped_image.save(os.path.join(save_directory, f"NPNs_{image_idx}_{idx}.png"))


if __name__ == "__main__":

    loop_through_dir = False

    pre_process = True
    crop_and_split = True
    threshold = False

    show_images = True
    save_images = True
    cell_only = True

    dataset_directory = os.path.join("..", "data", "asmm")
    image_directory = os.path.join(dataset_directory, "PEG100ImageScale", "set2")
    images_in_directory = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]

    save_directory = os.path.join(dataset_directory, "cell_imgs_npns", "cell_imgs_npcs")

    if loop_through_dir:
        for idx, image_path in enumerate(images_in_directory):
            process_image(
                image_idx=idx,
                image_path=image_path,
                crop_and_split=crop_and_split,
                threshold=threshold,
                show_images=show_images,
                save_images=save_images,
                save_directory=save_directory
            )
            print("Images saved.")
    else:
        process_image(
            image_idx=1,
            image_path=images_in_directory[0],
            crop_and_split=crop_and_split,
            threshold=threshold,
            show_images=show_images,
            save_directory=save_directory,
            cell_only=cell_only
        )
