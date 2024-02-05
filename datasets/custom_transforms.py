from typing import List
from torchvision import transforms
import torchvision.transforms.functional

# A dict listing all the transforms for datasets we might want to use.
transforms_dict = {
    "asmm-cell-npns": [
        transforms.ToPILImage(),
        # transforms.Resize([256, 256]),
        # transforms.RandomHorizontalFlip(0.1),
        # transforms.RandomVerticalFlip(0.1),
        # transforms.RandomRotation((5, 85)),
        # transforms.ToTensor(),
    ],
}


def image_transforms(dataset_name: str) -> transforms.Compose:
    if dataset_name not in transforms_dict.keys():
        raise Exception("Dataset not in list.")

    return transforms.Compose(transforms_dict[dataset_name])


def get_list_of_transforms() -> List[str]:
    return list(transforms_dict.keys())
