import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
object_path = "/data/img/sun.png"
background_folder = "/data/img/pictures"
target_folder = "/data/img/object_detection/"
checkpoint = "/data/model/net.pth"

scale = [0.25, 0.4]
num = [1,2,3]
img_size = 300
batch_size = 16
#epoch_lr = [(30,0.01),(30,0.001),(50,0.0001)]
epoch_lr = [(30,0.001)]