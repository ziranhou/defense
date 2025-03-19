import torch
import numpy as np


def evaluate_recall(rating, ground_truth, top_k):
    """
    计算推荐系统的召回率指标

    Args:
        rating (torch.Tensor): 模型输出的物品评分张量，形状为(n_items,)
        ground_truth (list/set): 用户真实交互的正样本索引集合
        top_k (int): 要考虑的推荐列表长度

    Returns:
        float: 召回率值，范围[0,1]
    """
    # 选取评分最高的top_k个物品
    _, rating_k = torch.topk(rating, top_k)
    # 将结果转移到CPU并转换为Python列表
    rating_k = rating_k.cpu().tolist()

    # 统计命中次数：预测结果中存在真实正样本的数量
    hit = 0
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            hit += 1

    # 计算召回率：命中数量占真实正样本总数的比例
    recall = hit / len(ground_truth)
    return recall



def evaluate_ndcg(rating, ground_truth, top_k):
    """
    计算NDCG(Normalized Discounted Cumulative Gain)评估指标

    Args:
        rating (torch.Tensor): 模型输出的评分/预测值张量，形状为[batch_size, item_count]
        ground_truth (list): 真实相关项的索引列表
        top_k (int): 要考虑的前k个推荐项

    Returns:
        float: 归一化折损累积增益值，范围在[0, 1]之间
    """
    # 获取预测评分中top_k个最高分的索引
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    dcg, idcg = 0., 0.

    # 同时计算DCG和IDCG
    for i, v in enumerate(rating_k):
        # 构建理想情况下的增益（假设前k个结果全部命中）
        if i < len(ground_truth):
            idcg += (1 / np.log2(2 + i))
        # 计算实际结果的折损增益
        if v in ground_truth:
            dcg += (1 / np.log2(2 + i))

    # 归一化处理：实际DCG / 理想DCG
    ndcg = dcg / idcg
    return ndcg

