import os
import numpy as np
import statistics


# Check the number of counts in the Cell Imgs Npns Labels

dataset_directory = os.path.join("..", "data", "asmm")
image_directory = os.path.join(dataset_directory, "cell_imgs_npns", "normal")
images_in_directory = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]

dmaps = [img for img in images_in_directory if img[-8:] == "dmap.npy"]
counts = [(np.sum(np.load(dmap))/100) for dmap in dmaps]

print(dmaps)
print(counts)

mean = statistics.mean(counts)
median = statistics.median(counts)

print(f"The mean number of particles is: {mean}. The median number is {median}")

