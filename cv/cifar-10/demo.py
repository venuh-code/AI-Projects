import torch
import os.path as osp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import vgg11, resnet18
from config import checkpoint_folder, label_list
from data import create_datasets

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def demo(img_path):
    totensor = transforms.ToTensor()
    img = Image.open(img_path).resize((32,32))

    img_tensor = totensor(img).unsqueeze(0)
    net = resnet18()
    net.load_state_dict(torch.load(osp.join(checkpoint_folder, "cls.pth")))
    net.eval()
    output = net(img_tensor)
    label = torch.argmax(output, dim=1)
    plt.imshow(np.array(img))
    print("it is", label)
    plt.title(str(label_list[label]))
    plt.show()

if __name__ == "__main__":
    demo("/data/img/206.png")