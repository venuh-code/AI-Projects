
import cv2
import numpy as np

size = 600
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((size,size,channels),np.uint8)
    sh=size/height
    sw=size/width
    for i in range(size):
        for j in range(size):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("/data/img/dog.jpg")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.imwrite("/data/img/dog_nst.jpg", zoom)
cv2.waitKey(0)

