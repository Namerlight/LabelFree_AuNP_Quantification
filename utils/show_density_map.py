import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sampledir = os.path.join("..", "data", "asmm", "data_vis_fig2")
dmaps_paths = [os.path.join(sampledir, curfile) for curfile in os.listdir(sampledir) if curfile[-3:] == "npy"]

print(dmaps_paths)

# for dp in dmaps_paths:
#     d_map_2d = np.load(dp)
#     d_map_2d = np.squeeze(d_map_2d)
#     plt.imshow(d_map_2d, cmap="turbo")
#     plt.imsave(dp+".png", d_map_2d)
#     plt.axis('off')
#     plt.show()


image_paths = [os.path.join(sampledir, curfile) for curfile in os.listdir(sampledir) if curfile[-3:] == "png"]

def open_image(path: str):
    return Image.open(path)

def threshold_image_fixed(img):
    ret, thresh4 = cv.threshold(img, 20, 255, cv.THRESH_TOZERO)
    images = [img, thresh4]
    return images[1]

for ip in image_paths:
    image = open_image(ip)
    opencvImage = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    thresholded_image = threshold_image_fixed(opencvImage)

    cv.imshow(ip, thresholded_image)
    cv.imwrite(ip+"thres.png", thresholded_image)
    k = cv.waitKey(0)
