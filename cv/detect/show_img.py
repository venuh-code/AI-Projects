from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import sys

#sys.path.append("..")

from generate_data import get_backgroud, combine_img
from config import (
    background_folder,
    object_path,
    scale,
    num,
    img_size,
    target_folder,
)

background_path = get_backgroud()
sun = Image.open(object_path).convert("RGB")
combined_img, box, _ = combine_img(background_path[10], sun)
img = Image.fromarray(combined_img)
draw = ImageDraw.Draw(img)
for b in box:
    cx,cy,w = b
    xmin = cx - w/2
    ymin = cy - w/2
    xmax = cx + w/2
    ymax = cy + w/2
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline = (0,0,255), width=5)
plt.imshow(img)
#plt.savefig("")
plt.show()
