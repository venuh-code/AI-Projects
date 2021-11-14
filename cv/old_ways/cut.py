import  numpy as np
import cv2
from matplotlib import pyplot as plt

def show(img):
    img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_)

img = cv2.imread(r"E:\task\data\sun.jpg")
show(img)

def grubcut(img,mask,rect,iters=20):
    img_ = img.copy()
    bg_model = np.zeros((1,65),np.float64)
    fg_model = np.zeros((1,65),np.float64)
    cv2.grabCut(img.copy(),mask,rect,bg_model,fg_model,iters,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    img_ = img * mask2[:,:, np.newaxis]
    return img_

mask = np.zeros(img.shape[:2], np.uint8)
rect = (250,60,700,400)
img_copy = img.copy()
cv2.rectangle(img_copy, rect[:2], rect[2:], (0,255,0), 3)
show(img_copy)
plt.show()
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = grubcut(img, mask, rect)
show(img)
plt.show()