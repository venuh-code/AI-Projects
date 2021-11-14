import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

def hash(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (8, 8))
    img = (img / 4).astype(np.uint8) * 4
    m = np.mean(img)
    img[img <= m] = 0
    img[img > m] = 1
    print(img.shape)
    plt.imshow(img * 255, cmap = "gray")
    return img.reshape(-1)

img1 = cv2.imread(r"E:\task\data\dog2.jpg", 0)
img2 = cv2.imread(r"E:\task\data\panda1.jpg", 0)
img3 = cv2.imread(r"E:\task\data\panda2.jpg", 0)

hash_img1 = hash(img1)
hash_img2 = hash(img2)
hash_img3 = hash(img3)

print(hash_img1 == hash_img2)


dis1 = np.sum(hash_img1 == hash_img2) / hash_img1.shape[0]
dis2 = np.sum(hash_img1 == hash_img3) / hash_img1.shape[0]

plt.subplot(131)
plt.xticks([])
plt.yticks([])
plt.imshow(img1)

plt.subplot(132)
plt.xticks([])
plt.yticks([])
plt.imshow(img2)
plt.title("dis: {}".format(dis1))

plt.subplot(133)
plt.xticks([])
plt.yticks([])
plt.imshow(img3)
plt.title("dis: {}".format(dis2))
plt.show()