import torchvision
from torchvision import transforms
import torch
from tqdm import tqdm

from config import epochs, device, data_folder, epochs, checkpoint_folder,batch_size
from model import vgg11

def create_datasets(data_folder, transform_train=None, transform_test=None):
    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    if transform_test  is None:
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    trainset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader

if __name__ == "__main__":
    trainloader, valloader = create_datasets(data_folder)


    net = vgg11().to(device)
    print(len(trainloader))
    for i, (img, label) in tqdm(
            enumerate(trainloader), total=len(trainloader)
    ):
        print(img.shape)