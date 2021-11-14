import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ld_find import lr_find
from model import resnet18
from data import create_datasets
from config import data_folder, batch_size, device, epochs
from generate_data import BoxData
from train_val import train_val

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    net = resnet18()
    net.linear = nn.Linear(in_features=512, out_features=4, bias=True)
    net.to(device)

    train_loader, val_loader = create_datasets(data_folder)
    traindata = BoxData(train_loader.dataset)
    trainloader = DataLoader(
        traindata, batch_size=batch_size, shuffle=True, num_workers=2
    )
    criteron = nn.L1Loss()
    valdata = BoxData(val_loader.dataset)
    valloader = DataLoader(
         valdata, batch_size=batch_size, shuffle=True, num_workers=2
    )

    best_lr = lr_find(net, optim.SGD, trainloader, criteron, model_name='reg')
    print("best_lr", best_lr)
    train_val(
        net, trainloader, valloader, criteron, epochs, device, model_name="reg"
   )

