import numpy as np
from glob import glob
import  os
from tqdm import tqdm
import re
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import  resize

img_path = sorted(glob(r"E:\task\data\English\Hnd\Img\*\*.png"))

def binary(img):
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            if img[i,j] < 0.5:
                img[i,j] = 0
            else:
                img[i,j] = 1
    return img

def preprocess(imng):
    width, height = img.shape
    rows, cols = np.where(img < 1.)
    print(img[151,454],img[0,0])
    print("rows:",rows)
    print("cols:", cols)

    x_min,x_max = min(rows), max(rows)
    y_min,y_max = min(cols), max(cols)
    size = max(y_max-y_min,x_max-x_min)

    x_ept = (size-(x_max-x_min))//2
    y_ept = (size-(y_max-y_min))//2

for img_path in tqdm(img_path):
    img = imread(img_path, as_grey=True)
    img = binary(img)
    preprocess(img)
    break