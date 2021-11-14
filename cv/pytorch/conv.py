from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
'''''''''
totensor = transforms.ToTensor()
toimg = transforms.ToPILImage()
net = resnet18(pretrained=True)
img = Image.open(r"E:\task\data\dog2.jpg")
img_tensor = totensor(img).unsqueeze(0)
img.show()
print(img_tensor.shape)
#print(net)

f1 = net.conv1(img_tensor)
print("f1 shape", f1.shape)
plt.figure(figsize=(10,10))
for p in range(4):
    f1_img_tensor = f1[0, p, :, :]
    f1_img = toimg(f1_img_tensor)
    plt.subplot(220+p+1)
    plt.imshow(f1_img)

plt.show()

f2 = net.bn1(f1)
plt.figure(figsize=(10,10))
for p in range(4):
    f2_img_tensor = f2[0,p,:,:]
    f2_img = toimg(f2_img_tensor)
    plt.subplot(220+p+1)
    plt.imshow(f2_img)
plt.show()

f3 = net.relu(f2)
plt.figure(figsize=(10,10))
for p in range(4):
    f3_img_tensor = f3[0,p,:,:]
    f3_img = toimg(f3_img_tensor)
    plt.subplot(220+p+1)
    plt.imshow(f3_img)
plt.show()

f4 = net.maxpool(f3)
plt.figure(figsize=(10,10))
for p in range(4):
    f4_img_tensor = f4[0,p,:,:]
    f4_img = toimg(f4_img_tensor)
    plt.subplot(220+p+1)
    plt.imshow(f4_img)
#plt.show()
'''
import numpy as np
'''''''''
x = np.linspace(0,1,100)
y = -np.log(1-x)
plt.plot(x,y)
plt.show()
'''''''''
x = np.linspace(-22,20,100)
def func(x):
    y = 2*x**2 + 3*x + 4
    return y
y = func(x)

def gradient(x):
    return 4*x + 3

def opt(x0=10, r = 0.6):
    x = x0
    path = []
    path.append(x)
    for i in range(10):
        x = x - r*gradient(x)
        print(x)
        path.append(x)
    return path

path = opt(-20)
plt.annotate("start", (-20,func(-20)))
plt.plot(x, y, color = 'g')
plt.plot(path, [func(x) for x in path], color = 'r')
plt.scatter(path,  [func(x) for x in path], color = 'r')
plt.show()