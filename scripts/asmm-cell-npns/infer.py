import os
from time import time
import numpy as np
import torchvision
import torch
import model_archs
import warnings
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode

warnings.filterwarnings("ignore", module="matplotlib\..*")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model_path = os.path.join("..", "..", "output", "models", "asmm-cell-npns_FCRNA_batch_8_epochs_101_02-37-54.pth")
# model = model_archs.FCRNA()

# model_path = os.path.join("..", "..", "output", "models", "asmm-cell-npns_MCNN_batch_8_epochs_101_19-32-33.pth")

model_path = os.path.join("..", "..", "output_ablation", "asmm-cell-npns_MCNN_batch_8_epochs_101_19-32-33.pth")
model = model_archs.MCNNog()
# model_path = os.path.join("..", "..", "output_ablation", "models", "asmm-cell-npns_MCNN_b1b2only_batch_16_epochs_80_lr_0.005_10-07-05.pth")
# model = model_archs.MCNN_b1b2only()

# model_path = os.path.join("..", "..", "output", "models", "asmm-cell-npns_UNet_batch_8_epochs_101_21-49-09.pth")
# model = model_archs.UNet(channels=3, filters=64, kernel_size=3)
# model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def threshold_image_fixed(img):
    ret, thresh4 = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)
    images = [img, thresh4]
    return images[1]


img_path = os.path.join("..", "..", "data", "asmm", "PEG100ImageScale", "48h")
img_path = os.path.join("..", "..", "data", "asmm", "PEG100ImageScale", "archived_set", "set4")
# img_path = os.path.join("..", "..", "data", "asmm", "cell_imgs_npns", "normal_val")
# real_label = os.path.join("..", "..", "data", "asmm", "cell_imgs_npns", "normal", "NPNs_4_9dmap.npy")

imgs = [os.path.join(img_path, img_file) for img_file in os.listdir(img_path) if img_file[-3:] == "png"]
print(imgs)

dmaps = [os.path.join(img_path, img_file) for img_file in os.listdir(img_path) if img_file[-8:] == "dmap.npy"]
print(dmaps)

dmaps = imgs

start = time()

num_objs_list, num_act_objs_list = [], []

for img_p, dmap_p in zip(imgs, dmaps):

    input_img = read_image(img_p, mode=ImageReadMode.RGB)
    imageObj = cv.imread(img_p)

    print("Calculating for:", img_p, "| dim:", imageObj.shape)

    # if imageObj.shape[0] <= 768 and imageObj.shape[1] <= 768:
    #     imageObj = cv.resize(imageObj, (1024, 1024),
    #                           interpolation=cv.INTER_CUBIC)
    #
    # if imageObj.shape[0] >= 1024 and imageObj.shape[1] >= 1024:
    #     if (imageObj.shape[0] >= 1536 and imageObj.shape[1] >= 1536):
    #         pass
    #     else:
    #         print("Resizing to smaller")
    #         imageObj = cv.resize(imageObj, (768, 768),
    #                           interpolation=cv.INTER_CUBIC)

    if imageObj.shape[0] <= 768 or imageObj.shape[1] <= 768:
        imageObj = cv.resize(imageObj, (1024, 1024),
                                  interpolation=cv.INTER_CUBIC)

    if imageObj.shape[0] >= 1536 or imageObj.shape[1] >= 1536:
        imageObj = cv.resize(imageObj, (1024, 1024),
                                  interpolation=cv.INTER_CUBIC)

    # to avoid grid lines
    # plt.axis("off")
    # plt.title("Original Image")
    # plt.imshow(cv.cvtColor(imageObj, cv.COLOR_RGB2BGR))
    # plt.show()
    # imageObj = threshold_image_fixed(imageObj)

    # plt.figure()
    # for channel_id, color in enumerate(["Red", "Green", "Blue"]):
    #     histogram, bin_edges = np.histogram(
    #         imageObj[:, :, channel_id], bins=256, range=(0, 256)
    #     )
    #     plt.plot(bin_edges[0:-1], histogram, color=color)

    # plt.title("Color Histogram")
    # plt.xlabel("Color value")
    # plt.ylabel("Pixel count")
    # plt.show()

    input_img = cv.cvtColor(imageObj, cv.COLOR_BGR2RGB)

    plt.imshow(input_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    transform = T.Compose([
        T.ToTensor(),
    ])

    current_inference = time()

    # print("Inference Input Shape:", (transform(input_img).to(device).float().unsqueeze(dim=0)).shape)

    density_map = model(transform(input_img).to(device).float().unsqueeze(dim=0))
    density_map = density_map.squeeze(dim=0).cpu().detach()
    current_end = time()

    # print("Time for inference:", current_end-current_inference)

    plt.imshow(density_map.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()

    # print(density_map.shape)

    plt.imshow(np.transpose(torchvision.utils.make_grid(density_map).detach().cpu().numpy(), (1, 2, 0)),
                                     interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()

    dmim = np.transpose(density_map, (1, 2, 0))
    plt.imshow(dmim, interpolation='nearest')
    plt.show()

    n_objects = torch.sum(density_map).item() / 100

    # act_objects = np.sum(np.load(dmap_p))/100
    act_objects = 1

    print(f"Image - {img_p.split(os.path.sep)[-1]} - The number of objects found: {n_objects}. Actual objects:", {act_objects})

    num_objs_list.append(int(abs(n_objects)))
    num_act_objs_list.append(int(abs(act_objects)))

print(num_objs_list, "Avg:", sum(num_objs_list)/len(num_objs_list))

num_objs_list = sorted(num_objs_list, reverse=True)[:5]

end = time()

print("Time taken", end-start)

# print("Count per image:", num_objs_list)

# avg = sum(num_objs_list)/len(num_objs_list)

# print("Average for this Incubation Time: ", avg)

# 6, 12, 18, 24, 48, 72
# real_values = [20, 60, 150, 375, 300, 280]
# gotten_values_with_MC-CNN = [53, 17, 80, 393, 188, 225]

# [20, 1, 2, 2, 1, 1, 1, 1, 2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 1, 11, 13, 18, 46, 3, 23, 22, 38, 116, 19, 4, 10, 12, 24, 80, 36, 6, 7, 16, 31, 1, 9, 3, 5, 12, 6, 5, 11, 7, 4, 19, 11, 22, 4]
# [35, 1, 2, 2, 2, 2, 4, 1, 2, 4, 5, 3, 4, 1, 4, 3, 3, 1, 1, 11, 36, 21, 20, 64, 5, 66, 43, 67, 193, 26, 13, 18, 20, 40, 132, 43, 13, 13, 23, 41, 2, 22, 6, 9, 16, 10, 9, 10, 13, 37, 21, 30, 12]