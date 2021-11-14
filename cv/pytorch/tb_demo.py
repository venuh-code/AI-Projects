from torchvision.models import vgg16
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

writer = SummaryWriter("log")
net = vgg16()
writer.add_graph(net, torch.randn((1,3,244,244)))
for i in range(100):
    writer.add_scalar("train/loss", (1000-i) * np.random.random(), i)
    writer.add_scalar("train/accuracy", i * np.random.random(), i)
writer.close()
