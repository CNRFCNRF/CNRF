import torch
from torch import nn


class GenerateSubnodes(nn.Module):
    def __init__(self, hidden_dim=4424):
        super(GenerateSubnodes, self).__init__()
        self.hidden_dim = hidden_dim
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

    def forward(self, proposals, up_feature, down_feature, head_boxes, tail_boxes, obj_embed, rel_pair_idxs):
        head_obj_embed = []
        tail_obj_embed = []
        for i, pair_idx in zip(obj_embed, rel_pair_idxs):
            head_obj_embed.append(i[pair_idx[:, 0]])
            tail_obj_embed.append(i[pair_idx[:, 1]])
        head_obj_embed = torch.cat(head_obj_embed, dim=0)
        tail_obj_embed = torch.cat(tail_obj_embed, dim=0)
        head_boxes = encode_box(head_boxes, proposals)
        head_boxes_embed = self.pos_embed(head_boxes)
        tail_boxes = encode_box(tail_boxes, proposals)
        tail_boxes_embed = self.pos_embed(tail_boxes)
        head_pre_rep = torch.cat((up_feature, head_obj_embed, head_boxes_embed), dim=-1)
        tail_pre_rep = torch.cat((down_feature, tail_obj_embed, tail_boxes_embed), dim=-1)
        return head_pre_rep, tail_pre_rep


def encode_box(box, proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for i, (proposal, boxes) in enumerate(zip(proposals, box)):
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                          w * h / (wid * hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)
