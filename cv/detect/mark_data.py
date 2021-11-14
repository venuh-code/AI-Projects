import torch
from torch.nn import CrossEntropyLoss, L1Loss

from config import img_size, device

def mark_data(boxes):
    label_matrix = torch.zeros((3,3)).to(device)
    offset_matrix = torch.ones((3,3,3)).to(device)
    confidences = torch.zeros((3,3)).to(device)

    grid_w = grid_h = img_size/3
    grids = torch.Tensor(
        [
            [100, 100, 100],
            [200, 100, 100],
            [300, 100, 100],
            [100, 200, 100],
            [200, 200, 100],
            [300, 200, 100],
            [100, 300, 100],
            [200, 300, 100],
            [300, 300, 100],
        ]
    )

    for box in boxes:
        cx,cy,w = box
        h = w
        #物体所在格子编号
        grid_x = int(cx/grid_w)
        grid_y = int(cy/grid_h)
        label_matrix[grid_y, grid_x] = 1
        offset_matrix[grid_y, grid_x] = torch.Tensor(
            [
                cx / ((grid_x * grid_w + grid_w)),
                cy / ((grid_y * grid_h + grid_h)),
                w / ((img_size)),
            ]
        )
        grid_box = grids[grid_x + 3 * grid_y]
        confidences[grid_y, grid_x] = iou(box, grid_box)

    return (
        label_matrix.view(-1,9) ,#  每一张图片9个格子
        offset_matrix.view(-1, 9, 3), # 每一个格子3个坐标
        confidences.view(-1, 9),
    )

class multi_box_loss(torch.nn.Module):
    def forward(
            self,
            label_prediction,
            offset_prediction,
            confidence_prediction,
            boxes_list,
    ):
        reg_criteron = L1Loss()
        label_tensor = []
        offset_tensor = []
        confidence_tensor = []

        for boxes in boxes_list:
            label, offset, confidence = mark_data(boxes)
            label_tensor.append(label)
            offset_tensor.append(offset)
            confidence_tensor.append(confidence)

        #一行一个标注框
        label_tensor = torch.cat(label_tensor, dim = 0).long()
        offset_tensor = torch.cat(offset_tensor, dim = 0)
        confidence_tensor = torch.cat(confidence_tensor, dim = 0)

        #添加掩码，负例不计算
        mask = label_tensor == 1  #torch.Size([16, 9])
        mask = mask.unsqueeze(2).float() #torch.Size([16, 9, 1])

        label_prediction = label_prediction.permute(0,2,1) #[batch, classsify, grid]
        weight = torch.Tensor([0.5, 1.5]).to(device)
        cls_criteron = CrossEntropyLoss(weight=weight.float())
        cls_loss = cls_criteron(label_prediction, label_tensor)

        offset_prediction = offset_prediction.view(-1,9,3) #[batch, grid, 3]
        reg_loss = reg_criteron(offset_prediction*mask , offset_tensor*mask)
        mask = mask.squeeze(2)
        confidence_loss = reg_criteron(
            confidence_prediction * mask, confidence_tensor * mask
        )
        return cls_loss + reg_loss + confidence_loss

def iou(box1, box2):
    cx_1, cy_1, w_1 = box1[:3]
    cx_2, cy_2, w_2 = box2[:3]

    xmin_1 = cx_1 - w_1 / 2
    ymin_1 = cy_1 - w_1 / 2
    xmax_1 = cx_1 + w_1 / 2
    ymax_1 = cy_1 + w_1 / 2

    xmin_2 = cx_2 - w_2 / 2
    ymin_2 = cy_2 - w_2 / 2
    xmax_2 = cx_2 + w_2 / 2
    ymax_2 = cy_2 + w_2 / 2

    if (
        ymax_1 <= ymin_2
        or ymax_2 <= ymin_1
        or xmax_2 <= xmin_1
        or xmax_1 <= xmin_2
    ):
        return 0.0

    inter_x_min = max(xmin_1, xmin_2)
    inter_y_min = max(ymin_1, ymin_2)
    inter_x_max = min(xmax_1, xmax_2)
    inter_y_max = min(ymax_1, ymax_2)

    intersection = (inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)
    return intersection / (w_1 * w_1 + w_2 * w_2)



