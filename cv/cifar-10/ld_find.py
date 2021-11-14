import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import device, data_folder
from model import vgg11, resnet18
from data import create_datasets

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def lr_find(
        net,
        optimizer_class,
        dataloader,
        criteron,
        lr_list=[1*10**(i/2) for i in range(-20, 0)],
        show=False,
        test_times=10,
        model_name='cls',
        ):

    params = net.state_dict().copy()
    loss_matrix = []
    for i ,(img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        loss_list = []
        k =0
        for lr in tqdm(lr_list):
            net.load_state_dict(params)
            print("img shape", img.shape)
            out = net(img)
            optimizer = optimizer_class(
                net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )

            loss = criteron(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            new_out = net(img)
            new_loss = criteron(new_out, label)
            if k == 0:
                if model_name == 'cls':
                    pred = torch.argmax(new_out, dim=1)
                    acc = torch.sum(pred == label)
                else:
                    acc = torch.sum(new_out == label)
                print("acc is ", acc/128)
            k = k+1
            loss_list.append(new_loss.item())
        loss_matrix.append(loss_list)
        if i + 1 == test_times:
            break
    loss_matrix = np.array(loss_matrix)
    loss_matrix = np.mean(loss_matrix, axis=0)
    if show:
        plt.plot([np.log10(lr) for lr in lr_list], loss_matrix)
        plt.savefig("/data/img/lr_find.jpg")

    decrease = [
        loss_matrix[i+1] - loss_matrix[i] for i in range(len(lr_list) - 1)
    ]

    max_decease = np.argmin(decrease)
    best_lr = lr_list[max_decease]
    print("best lr is:", best_lr)
    return best_lr

if __name__ == "__main__":
    net = resnet18().to(device)
    trainloader, _ = create_datasets(data_folder)
    criteron = CrossEntropyLoss()
    lr_list = [1*10 ** (i/3) for i in range(-30, 0)]
    lr_find(net, SGD, trainloader, criteron, show=True)

