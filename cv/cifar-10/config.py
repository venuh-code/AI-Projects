import torch

device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

data_folder = "/data/"
checkpoint_folder = "/data/"
batch_size = 256
epochs = [(10, 0.01), (5, 0.001), (3, 0.0001)]
#epochs = [(30,0.0001), (10 ,0.00001)]

label_list = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship" ,"trunk"
]