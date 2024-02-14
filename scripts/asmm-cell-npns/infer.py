import os
from time import time
import numpy as np
import torchvision
import torch
import model_archs
import warnings
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode

warnings.filterwarnings("ignore", module="matplotlib\..*")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = os.path.join("..", "..", "output_ablation", "asmm-cell-npns_MCNN_batch_8_epochs_101_19-32-33.pth")
model = model_archs.MCNNog()

model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

def threshold_image_fixed(img):
    ret, thresh4 = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)
    images = [img, thresh4]
    return images[1]


img_path = os.path.join("..", "..", "data", "asmm", "PEG100ImageScale", "48h")

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

    if imageObj.shape[0] <= 768 or imageObj.shape[1] <= 768:
        imageObj = cv.resize(imageObj, (1024, 1024),
                                  interpolation=cv.INTER_CUBIC)

    if imageObj.shape[0] >= 1536 or imageObj.shape[1] >= 1536:
        imageObj = cv.resize(imageObj, (1024, 1024),
                                  interpolation=cv.INTER_CUBIC)

    input_img = cv.cvtColor(imageObj, cv.COLOR_BGR2RGB)

    plt.imshow(input_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    transform = T.Compose([
        T.ToTensor(),
    ])

    current_inference = time()

    density_map = model(transform(input_img).to(device).float().unsqueeze(dim=0))
    density_map = density_map.squeeze(dim=0).cpu().detach()
    current_end = time()

    plt.imshow(density_map.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()

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