import torchvision
from torchvision import transforms
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from config import data_folder


def show_batch(display_transform=None):
    if display_transform is None:
        display_transform = transforms.ToTensor()
    display_set = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=display_transform
    )
    display_loader = torch.utils.data.DataLoader(display_set, batch_size=32)
    topil = transforms.ToPILImage()
    i = 0
    for batch_img, batch_label in display_loader:
        grid = make_grid(batch_img, nrow = 8)
        grid_img = topil(grid)
        plt.figure(figsize=(15,15))
        plt.imshow(grid_img)
        grid_img.save("trans_cifar10.png")

        plt.show()
        break


if __name__ == "__main__":
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    show_batch(transform_train)
