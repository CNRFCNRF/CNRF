import torch


def generate_reassignment_labels(global_rel_dists):
    global_shape = global_rel_dists.shape[0]
    hundred_min = -100 * torch.ones(global_shape, 1).to(torch.device("cuda"))
    global_rel_dists_50 = global_rel_dists[:, 1:]
    global_rel_dists_51 = torch.cat((hundred_min, global_rel_dists_50), dim=-1)
    scores, id = torch.sort(global_rel_dists_51, dim=-1, descending=True)
    global_pred_score = scores[:, 0:5]
    rel_rank_id = torch.tensor([0, 8, 44, 34, 28, 30, 17, 16, 10, 32, 31, 22, 40, 26, 23, 45, 20, 36,
                                49, 18, 2, 9, 3, 15, 27, 35, 48, 39, 41, 6, 4, 1, 47, 19, 33, 37,
                                43, 25, 21, 42, 12, 14, 38, 11, 46, 50, 24, 29, 5, 13, 7]).to(torch.device("cuda"))
    zeros = torch.zeros(global_pred_score.shape[0], 1).to(torch.device("cuda"))
    ones = torch.ones(global_pred_score.shape[0], 1).to(torch.device("cuda"))
    zero_padding = torch.zeros((global_pred_score.shape[0], 46)).to(torch.device("cuda"))
    reassignment_score_1 = torch.unsqueeze(global_pred_score[:, 0], dim=1)
    relevant_scores = global_pred_score / reassignment_score_1
    reassignment_score = reassignment_score_1 / global_pred_score
    id_1 = id[:, 0].unsqueeze(dim=1)
    id_2 = id[:, 1].unsqueeze(dim=1)
    id_3 = id[:, 2].unsqueeze(dim=1)
    id_4 = id[:, 3].unsqueeze(dim=1)
    id_5 = id[:, 4].unsqueeze(dim=1)
    reassignment_score_1 = reassignment_score[:, 0].unsqueeze(dim=1)
    reassignment_score_2 = reassignment_score[:, 1].unsqueeze(dim=1)
    reassignment_score_3 = reassignment_score[:, 2].unsqueeze(dim=1)
    reassignment_score_4 = reassignment_score[:, 3].unsqueeze(dim=1)
    reassignment_score_5 = reassignment_score[:, 4].unsqueeze(dim=1)

    id1_rank = rel_rank_id[id_1]
    id2_rank = rel_rank_id[id_2]
    id3_rank = rel_rank_id[id_3]
    id4_rank = rel_rank_id[id_4]
    id5_rank = rel_rank_id[id_5]
    relative_rank = torch.max(id1_rank, id2_rank)
    reassignment_score_1 = torch.where(id1_rank >= relative_rank, reassignment_score_1, zeros)
    reassignment_score_2 = torch.where(id2_rank >= relative_rank, reassignment_score_2, zeros)
    relative_rank = torch.max(relative_rank, id3_rank)
    reassignment_score_3 = torch.where(id3_rank >= relative_rank, reassignment_score_3, zeros)
    relative_rank = torch.max(relative_rank, id4_rank)
    reassignment_score_4 = torch.where(id4_rank >= relative_rank, reassignment_score_4, zeros)
    relative_rank = torch.max(relative_rank, id5_rank)
    reassignment_score_5 = torch.where(id5_rank >= relative_rank, reassignment_score_5, zeros)

    reassignment_labels = torch.cat(
        (reassignment_score_1, reassignment_score_2, reassignment_score_3, reassignment_score_4, reassignment_score_5),
        dim=-1)
    reassignment_labels = torch.where(relevant_scores < 0.3, zeros, reassignment_labels)

    reassignment_labels = torch.cat((reassignment_labels, zero_padding), dim=1)
    _, restore_id = torch.sort(id, dim=1, descending=True)
    reassignment_labels = reassignment_labels.gather(dim=1, index=restore_id)

    return reassignment_labels
