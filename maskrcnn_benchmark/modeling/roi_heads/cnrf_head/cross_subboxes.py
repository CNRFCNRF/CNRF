import torch
from maskrcnn_benchmark.modeling.roi_heads.cnrf_head.boxes_coordinates import generate_boxes_coordinates


def generate_cross_subboxes(proposals, rel_pair_idxs):
    condition_1, condition_2, condition_3, condition_4, \
    inter_x1, inter_y1, inter_x2, inter_y2, \
    h_x_1, h_y_1, h_x_2, h_y_2, t_x_1, t_y_1, t_x_2, t_y_2, \
    center_h_x, center_h_y, center_t_x, center_t_y, num_rels \
        = generate_boxes_coordinates(proposals, rel_pair_idxs)
    y_where = (h_y_2 - h_y_1) / 2 - (t_y_2 - t_y_1) / 2
    num_rel = condition_1.shape[0]
    ones = torch.ones(num_rel, 1).to(torch.device('cuda'))
    zeros = torch.zeros(num_rel, 1).to(torch.device('cuda'))

    interaction_matrix = torch.where(condition_1 <= 0, zeros, ones)
    interaction_matrix = torch.where(condition_2 <= 0, zeros, interaction_matrix)
    '''Horizontal interaction'''
    horizontal_matrix = torch.where(condition_3 <= 0, interaction_matrix, zeros)
    horizontal_matrix = torch.where(condition_4 > 0, horizontal_matrix, zeros)
    '''Longitudinal interaction'''
    longitudinal_matrix1 = torch.where(condition_3 > 0, interaction_matrix, zeros)
    longitudinal_matrix1 = torch.where(condition_4 <= 0, longitudinal_matrix1, zeros)

    longitudinal_matrix2 = torch.where(condition_3 > 0, interaction_matrix, zeros)
    longitudinal_matrix2 = torch.where(condition_4 > 0, longitudinal_matrix2, zeros)
    longitudinal2_judgment = abs(inter_x1 - inter_x2) - abs(inter_y1 - inter_y2)
    longitudinal2_h_matrix = torch.where(longitudinal2_judgment > 0, longitudinal_matrix2, zeros)
    longitudinal2_l_matrix = torch.where(longitudinal2_judgment <= 0, longitudinal_matrix2, zeros)

    '''Subset'''
    subset_matrix = torch.where(condition_3 <= 0, interaction_matrix, zeros)
    subset_matrix = torch.where(condition_4 <= 0, subset_matrix, zeros)

    subset_up_matrix = torch.where(y_where <= 0, subset_matrix, zeros)
    subset_down_matrix = torch.where(y_where > 0, subset_matrix, zeros)

    '''generate cross_subboxes'''

    horizontal_boxes_head = horizontal_matrix * torch.cat((torch.min(inter_x1, h_x_1), torch.min(inter_y1, center_h_y),
                                                           torch.max(inter_x2, h_x_2), torch.max(inter_y2, center_h_y)),
                                                          dim=-1)
    horizontal_boxes_tail = horizontal_matrix * torch.cat((torch.min(inter_x1, t_x_1), torch.min(inter_y1, center_t_y),
                                                           torch.max(inter_x2, t_x_2), torch.max(inter_y2, center_t_y)),
                                                          dim=-1)

    longitudinal_boxes_head = longitudinal_matrix1 * torch.cat(
        (torch.min(inter_x1, center_h_x), torch.min(inter_y1, h_y_1),
         torch.max(inter_x2, center_h_x), torch.max(inter_y2, h_y_2)),
        dim=-1)
    longitudinal_boxes_tail = longitudinal_matrix1 * torch.cat(
        (torch.min(inter_x1, center_t_x), torch.min(inter_y1, t_y_1),
         torch.max(inter_x2, center_t_x), torch.max(inter_y2, t_y_2)),
        dim=-1)

    longitudinal2_h_boxes_head = longitudinal2_h_matrix * torch.cat(
        (torch.min(inter_x1, h_x_1), torch.min(inter_y1, center_h_y),
         torch.max(inter_x2, h_x_2), torch.max(inter_y2, center_h_y)),
        dim=-1)
    longitudinal2_h_boxes_tail = longitudinal2_h_matrix * torch.cat(
        (torch.min(inter_x1, t_x_1), torch.min(inter_y1, center_t_y),
         torch.max(inter_x2, t_x_2), torch.max(inter_y2, center_t_y)),
        dim=-1)
    longitudinal2_l_boxes_head = longitudinal2_l_matrix * torch.cat(
        (torch.min(inter_x1, center_h_x), torch.min(inter_y1, h_y_1),
         torch.max(inter_x2, center_h_x), torch.max(inter_y2, h_y_2)),
        dim=-1)
    longitudinal2_l_boxes_tail = longitudinal2_l_matrix * torch.cat(
        (torch.min(inter_x1, center_t_x), torch.min(inter_y1, t_y_1),
         torch.max(inter_x2, center_t_x), torch.max(inter_y2, t_y_2)),
        dim=-1)

    subset_up_boxes_head = subset_up_matrix * torch.cat(
        (torch.min(inter_x1, h_x_1), torch.min(inter_y1, center_h_y),
         torch.max(inter_x2, h_x_2), torch.max(inter_y2, h_y_2)), dim=-1)
    subset_up_boxes_tail = subset_up_matrix * torch.cat((torch.min(inter_x1, t_x_1), torch.min(inter_y1, t_y_1),
                                                         torch.max(inter_x2, t_x_2),
                                                         torch.max(inter_y2, center_t_y)), dim=-1)

    subset_down_boxes_head = subset_down_matrix * torch.cat((torch.min(inter_x1, h_x_1), torch.min(inter_y1, h_y_1),
                                                             torch.max(inter_x2, h_x_2),
                                                             torch.max(inter_y2, center_h_y)), dim=-1)
    subset_down_boxes_tail = subset_down_matrix * torch.cat(
        (torch.min(inter_x1, t_x_1), torch.min(inter_y1, center_t_y),
         torch.max(inter_x2, t_x_2), torch.max(inter_y2, t_y_2)), dim=-1)

    interaction_h_boxes = horizontal_boxes_head + longitudinal_boxes_head + longitudinal2_h_boxes_head + longitudinal2_l_boxes_head + \
                          subset_up_boxes_head + subset_down_boxes_head
    interaction_t_boxes = horizontal_boxes_tail + longitudinal_boxes_tail + longitudinal2_h_boxes_tail + longitudinal2_l_boxes_tail + \
                          subset_up_boxes_tail + subset_down_boxes_tail

    non_interaction_h_boxes, non_interaction_t_boxes = generate_non_interaction_boxes(condition_1, condition_2, zeros,
                                                                                      interaction_matrix,
                                                                                      h_x_1, h_y_1, h_x_2, h_y_2,
                                                                                      t_x_1, t_y_1, t_x_2, t_y_2,
                                                                                      center_h_x, center_h_y,
                                                                                      center_t_x, center_t_y)
    cross_head_boxes = interaction_h_boxes + non_interaction_h_boxes
    cross_tail_boxes = interaction_t_boxes + non_interaction_t_boxes
    cross_head_boxes = cross_head_boxes.split(num_rels, dim=0)
    cross_tail_boxes = cross_tail_boxes.split(num_rels, dim=0)

    return cross_head_boxes, cross_tail_boxes, interaction_matrix


def generate_non_interaction_boxes(condition_1, condition_2, zeros, youjiaoji_matrix,
                                   x1, y1, x2, y2, x3, y3, x4, y4, center_head_x, center_head_y, center_tail_x,
                                   center_tail_y):
    non_interaction_matrix = -1 * (youjiaoji_matrix - 1)
    """No longitudinal interaction"""
    l_no_interaction_matrix = torch.where(condition_1 > 0, non_interaction_matrix, zeros)
    l_no_interaction_matrix = torch.where(condition_2 <= 0, l_no_interaction_matrix, zeros)
    """No horizontal interaction"""
    h_no_interaction_matrix = torch.where(condition_1 <= 0, non_interaction_matrix, zeros)
    h_no_interaction_matrix = torch.where(condition_2 > 0, h_no_interaction_matrix, zeros)
    """No interaction"""
    no_interaction_matrix = torch.where(condition_1 <= 0, non_interaction_matrix, zeros)
    no_interaction_matrix = torch.where(condition_2 <= 0, no_interaction_matrix, zeros)

    l_no_interaction_boxes_head = l_no_interaction_matrix * \
                                  torch.cat((torch.min(x1, center_head_x), torch.min(y4, center_head_y),
                                             torch.max(x2, center_head_x), torch.max(y3, center_head_y)), dim=-1)
    l_no_interaction_boxes_tail = l_no_interaction_matrix * \
                                  torch.cat((torch.min(x3, center_tail_x), torch.min(y2, center_tail_y),
                                             torch.max(x4, center_tail_x), torch.max(y1, center_tail_y)), dim=-1)

    h_no_interaction_boxes_head = h_no_interaction_matrix * \
                                  torch.cat((torch.min(x4, center_head_x), torch.min(y1, center_head_y),
                                             torch.max(x3, center_head_x), torch.max(y2, center_head_y)), dim=-1)
    h_no_interaction_boxes_tail = h_no_interaction_matrix * \
                                  torch.cat((torch.min(x2, center_tail_x), torch.min(y3, center_tail_y),
                                             torch.max(x1, center_tail_x), torch.max(y4, center_tail_y)), dim=-1)

    no_interaction_head = no_interaction_matrix * torch.cat((torch.min(x4, center_head_x), torch.min(y4, center_head_y),
                                                             torch.max(x3, center_head_x),
                                                             torch.max(y3, center_head_y)),
                                                            dim=-1)
    no_interaction_tail = no_interaction_matrix * torch.cat((torch.min(x2, center_tail_x), torch.min(y2, center_tail_y),
                                                             torch.max(x1, center_tail_x),
                                                             torch.max(y1, center_tail_y)),
                                                            dim=-1)

    non_interaction_h_boxes = l_no_interaction_boxes_head + h_no_interaction_boxes_head + no_interaction_head
    non_interaction_t_boxes = l_no_interaction_boxes_tail + h_no_interaction_boxes_tail + no_interaction_tail

    return non_interaction_h_boxes, non_interaction_t_boxes
