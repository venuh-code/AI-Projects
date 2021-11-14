from torch import optim, nn
import torch
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import epochs, device, data_folder, epochs, checkpoint_folder
from data import create_datasets
from model import vgg11, resnet18

def train_val(
        net, trainloader, valloader, criteron, epochs, device, model_name="cls"
):
    best_acc = 0.0
    best_loss = 1e9
    writer = SummaryWriter("log")
    if osp.exists(osp.join(checkpoint_folder, model_name + ".pth")):
        net.load_state_dict(torch.load(osp.join(checkpoint_folder, model_name + ".pth")))
        print("load model ok")
    else:
        pass

    for n, (num_epochs, lr) in enumerate(epochs):
        optimier = optim.SGD(
            net.parameters(), lr=lr, weight_decay=5e-2, momentum=0.9,
        )
        print("Training:",num_epochs, lr)
        for epoch in range(num_epochs):
            net.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            #for i, (img, label) in tqdm(
                #enumerate(trainloader), total=len(trainloader)
           # ):

            for img, label in trainloader:
                img, label = img.to(device), label.to(device)
                output = net(img)
                optimier.zero_grad()
                loss = criteron(output, label)
                loss.backward()
                optimier.step()

                if model_name == 'cls':
                    pred = torch.argmax(output, dim=1)
                    acc = torch.sum(pred==label)
                    epoch_acc += acc.item()
                epoch_loss += loss.item() * img.shape[0]
            epoch_loss /= len(trainloader.dataset)

            if model_name == 'cls':
                epoch_acc /= len(trainloader.dataset)
                print(
                    "epoch loss: {:.8f} epoch accuracy : {:.8f}".format(epoch_loss, epoch_acc)
                )
                writer.add_scalar(
                    "epoch_loss_{}".format(model_name),
                    epoch_loss,
                    sum([e[0] for e in epochs[:n]]) + epoch,
                )
                writer.add_scalar(
                    "epoch_acc_{}".format(model_name),
                    epoch_acc,
                    sum([e[0] for e in epochs[:n]]) + epoch,
                )
            else:
                print("epoch loss: {:.8f}".format(epoch_loss))
                writer.add_scalar(
                    "epoch_loss_{}".format(model_name),
                    epoch_loss,
                    sum([e[0] for e in epochs[:n]]) + epoch,
                )

            #在无梯度模式下快速验证
            with torch.no_grad():
                net.eval()
                val_loss = 0.0
                val_acc = 0.0
                for i, (img, label) in tqdm(
                    enumerate(valloader), total=len(valloader)
                ):
                    img, label = img.to(device), label.to(device)
                    output = net(img)
                    loss = criteron(output, label)
                    if model_name == 'cls':
                        pred = torch.argmax(output, dim = 1)
                        acc = torch.sum(pred == label)
                        val_acc += acc.item()
                    val_loss += loss.item() * img.shape[0]
                val_loss /= len(valloader.dataset)
                val_acc /= len(valloader.dataset)

                if model_name == 'cls':
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(
                            net.state_dict(),
                            osp.join(checkpoint_folder, model_name + ".pth")
                        )
                    print(
                        "validation loss: {:.8f} validation accuracy : {:.8f}".format(
                            val_loss, val_acc)
                    )

                    writer.add_scalar(
                        "validation_loss_{}".format(model_name),
                        val_loss,
                        sum([e[0] for e in epochs[:n]]) + epoch,
                    )
                    writer.add_scalar(
                        "validation_acc_{}".format(model_name),
                        val_acc,
                        sum([e[0] for e in epochs[:n]]) + epoch,
                    )
                else:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(
                            net.state_dict(),
                            osp.join(checkpoint_folder, model_name),
                        )
                    print(
                        "validation loss: {:.8f}".format(val_loss)
                    )
                    writer.add_scalar(
                        "validation_loss_{}".format(model_name),
                        val_loss,
                        sum([e[0] for e in epochs[:n]]) + epoch,
                    )
    writer.close()

if __name__ == "__main__":
    trainloader, valloader = create_datasets(data_folder)
    net = resnet18().to(device)#vgg11().to(device)
    criteron = nn.CrossEntropyLoss()
    train_val(net, trainloader, valloader, criteron, epochs, device)