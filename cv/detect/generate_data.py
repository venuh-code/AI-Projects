from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path as osp
import  os
import re
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from config import (
    background_folder,
    object_path,
    scale,
    num,
    img_size,
    target_folder,
)

def get_backgroud():
    background_paths = glob(osp.join(background_folder, "*.jpg"))

    # for i in background_paths:
    #     img = Image.open(i).convert("RGB").resize((300,300))
    #     plt.imshow(img)
    #     plt.show()
    #     break

    return background_paths

def extract_sun(sun):
    sun = np.array(sun)
    return np.where(np.mean(sun,axis=2) < 250)

def combine_img(background_path, sun):
    sun_num = np.random.choice(num)
    background = np.array(
        Image.open(background_path).convert("RGB").resize((300,300))
    )
    location = []
    coordinates = []
    for n in range(sun_num):
        located = False
        while not located:
            s = np.random.random() * (scale[1] - scale[0]) + scale[0]
            sun_size = int(img_size * s)
            sun = sun.resize((sun_size, sun_size))
            single_sun = extract_sun(sun)
            cx = np.random.random() * img_size
            cy = np.random.random() * img_size
            if (
                cx + sun_size / 2 >= img_size
                or cy + sun_size / 2 >= img_size
                or cx - sun_size / 2 < 0
                or cy - sun_size / 2 < 0
            ):
                continue
            overlap = False
            for loc in location:
                p_sun_size = loc[2]
                p1x = loc[0] - p_sun_size / 2
                p1y = loc[1] - p_sun_size / 2
                p2x = loc[0] + p_sun_size / 2
                p2y = loc[1] + p_sun_size / 2
                p3x = cx - sun_size / 2
                p3y = cy - sun_size / 2
                p4x = cx + sun_size / 2
                p4y = cy + sun_size / 2
                if (p1y < p4y) and (p3y < p2y) and (p1x < p4x) and (p2x > p3x):
                    overlap = True
                    break

            if overlap:
                continue
            located = True
            location.append((int(cx), int(cy), sun_size))

        sun_coords_x = single_sun[0] + int(cy - sun_size / 2)
        sun_coords_y = single_sun[1] + int(cx - sun_size / 2)
        sun_coords = tuple((sun_coords_x, sun_coords_y))
        background[sun_coords] = np.array(sun)[single_sun]
        coordinates.append(sun_coords)

    return background, location, coordinates

def generate_data():
    backgroud_paths = get_backgroud()
    sun = Image.open(object_path).convert("RGB")
    if not osp.exists(target_folder):
        os.makedirs(target_folder)
    segmentation_folder= re.sub(
        "object_detection\/$", "segmentation", target_folder
    )
    if not osp.exists(segmentation_folder):
        os.makedirs(segmentation_folder)
    for i, item in tqdm(
        enumerate(backgroud_paths), total=len(backgroud_paths)
    ):
        combined_img, loc, coord = combine_img(item, sun)
        target_path = osp.join(target_folder, "{:0>3d}.jpg". format(i))
        plt.imsave(target_path, combined_img)
        with open(re.sub(".jpg", ".txt", target_path), "w") as f:
            f.write(str(loc))
        mask = np.zeros((img_size, img_size, 3))
        for c in coord:
            mask[c] = 1
        segmentation_path = osp.join(
            segmentation_folder, "{:0>3d}.jpg".format(i)
        )

        plt.imsave(segmentation_path, mask)

if __name__ == '__main__':
    generate_data()
