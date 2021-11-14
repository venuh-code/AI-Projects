import torch
from torch import nn
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from PIL import ImageDraw
import numpy as np

from data import DetectionData, TestTransform
from config import checkpoint

def py_cpu_nms(boxes, scores, thresh): #boxes = [xmin.item(), ymin.item(), xmax.item(), ymax.item()]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #print(scores) #[0.15951303 0.22561514 0.18636191]
    order = scores.argsort()[::-1] #duo ge grid[0.9, 0.8,0.4,0.8,0.2]
    #print(order) [1 2 0]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  #>> np.maximum([-2, -1, 0, 1, 2], 0)==>array([0, 0, 0, 1, 2])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

dataset = DetectionData(subset="test", transform=TestTransform())
net = resnet18()
net.fc = nn.Linear(512, 54)
net.load_state_dict(torch.load(checkpoint))
net.eval()

img, boxes = dataset[49]

#predice
out = net(img.unsqueeze(0))
out_label = out[:, :18].view(-1, 9, 2)
out_offset = out[:, 18:45]
out_score = out[:, 45:]

predict_label = torch.argmax(out_label, dim = 2)

predict_offset = out_offset.view(-1, 9, 3)
anchors = torch.tensor(
    [
        [100, 100, 300],
        [200, 100, 300],
        [300, 100, 300],
        [100, 200, 300],
        [200, 200, 300],
        [300, 200, 300],
        [100, 300, 300],
        [200, 300, 300],
        [300, 300, 300],
    ]
)

predict_box = predict_offset * anchors
topil = ToPILImage()
img_pil = topil(img)
img_pil_nms = img_pil.copy()
draw = ImageDraw.Draw(img_pil)
positive_boxes = []
positive_scores = []
for i, b in enumerate(predict_box[0]):
    if predict_label[0][i] == 1:
        xmin = b[0] - b[2] / 2
        ymin = b[1] - b[2] / 2
        xmax = b[0] + b[2] / 2
        ymax = b[1] + b[2] / 2
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0,255,0))
        draw.text((xmin, ymin), "{},{}".format(out_score[:, i].item(), 49))
        positive_boxes.append(
            [xmin.item(), ymin.item(), xmax.item(), ymax.item()]
        )
        positive_scores.append(out_score[:, i].item())
plt.figure(figsize=(5,5))
plt.imshow(img_pil)

draw_nms = ImageDraw.Draw(img_pil_nms)
boxes = np.array(positive_boxes)
scores = np.array(positive_scores)
keep_idx = py_cpu_nms(boxes, scores, 0.3)
keep_box = boxes[keep_idx]
for i, b in enumerate(keep_box):
    xmin, ymin, xmax, ymax = b
    draw_nms.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0))
    draw_nms.text((xmin, ymin), "{}".format(scores[i].item()))
plt.figure(figsize=(5,5))
plt.imshow(img_pil_nms)

plt.show()
