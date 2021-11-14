import torch
from torch.utils.data import Dataset
from torchvision import transforms
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from data import create_datasets
from config import data_folder

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def expand(img, background=(128,128,128), show=False):
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    img = np.array(topil(img)).astype(np.uint8)
    height, width, depth = img.shape
    ratio = random.uniform(1,2)
    left = random.uniform(0.3 * width, width * ratio - width)
    top = random.uniform(0.3 * width, width * ratio - width)

    while int(left + width) > int(width * ratio) or int(top + height) > int(height * ratio):
        ratio = random.uniform(1, 2)
        left = random.uniform(0.3 * width, width * ratio - width)
        top = random.uniform(0.3 * width, width * ratio - width)
    expand_img = np.zeros(
        (int(height * ratio), int(width * ratio), depth), dtype = img.dtype
    )

    expand_img[:, :, :] = background
    expand_img[
        int(top) : int(top+height), int(left) : int(left+width)
    ] = img

    if show:
        expand_img_ = Image.fromarray(expand_img)
        draw = ImageDraw.ImageDraw(expand_img_)
        draw.rectangle(
            [(int(left), int(top)), (int(left + width), int(top + height))],
            outline=(0, 255, 0),
            width=2,
        )
        expand_img_.save("/data/img/plane.jpg")
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(expand_img_)
        plt.savefig("/data/img/expand.jpg")
        plt.show()

    xmin = left / (width * ratio)
    ymin = top / (height * ratio)
    xmax = (left + width) / (width * ratio)
    ymax = (top + height) / (height * ratio)

    expand_img = totensor(
        Image.fromarray(expand_img).resize((32,32), Image.BILINEAR)
    )

    return expand_img, torch.Tensor([xmin, ymin, xmax, ymax])

class BoxData(Dataset):
    def __init__(self, dataset, show=False):
        super(BoxData, self).__init__()
        self.dataset = dataset
        self.show = show

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img, box = expand(img, show=self.show)
        return img, box

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader, _ = create_datasets(data_folder, transform_train=transform)
    data = BoxData(train_loader.dataset, show=False)
    print(data[0][0].shape, data[0][1].shape)