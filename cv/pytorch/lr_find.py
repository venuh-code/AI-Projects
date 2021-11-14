import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import device, data_folder
from model import vgg11
