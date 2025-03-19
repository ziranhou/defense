import torch
import numpy as np
import torch.nn as nn
from parse import args
from eval import evaluate_recall, evaluate_ndcg


class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        """初始化数据集的划分和嵌入层
        参数说明：
        train_ind: list/array, 训练集的样本索引列表
        test_ind: list/array, 测试集的样本索引列表
        target_ind: list/array, 候选目标样本索引列表
        m_item: int, 项目总数（用于生成负样本）
        dim: int, 嵌入向量的维度

        功能说明：
        1. 过滤target_ind中已存在于train/test集合的样本
        2. 创建训练集的负样本集合
        3. 初始化用户嵌入层"""
        super().__init__()  # 调用父类构造函数

        # 初始化基础数据集
        self._train_ = train_ind
        self._test_ = test_ind
        # 过滤目标样本：排除已出现在训练/测试集中的样本
        self._target_ = []
        for i in target_ind:
            if i in train_ind or i in test_ind:
                continue
            self._target_.append(i)
        self.m_item = m_item

        # 转换为PyTorch张量
        self._train = torch.Tensor(train_ind).long()
        self._test = torch.Tensor(test_ind).long()

        # 生成训练负样本：确保负样本不在训练集中
        train_neg_ind = []
        for _ in train_ind:
            neg_item = np.random.randint(m_item)
            while neg_item in train_ind:  # 排除训练集中已有的样本
                neg_item = np.random.randint(m_item)
            train_neg_ind.append(neg_item)

        # 合并正负样本并建立索引映射
        train_all = train_ind + train_neg_ind
        train_all.sort()
        self.train_all = torch.Tensor(train_all).long()

        # 创建正负样本位置索引
        d = {idx: train_all_idx for train_all_idx, idx in enumerate(train_all)}
        self._train_pos = torch.Tensor([d[i] for i in train_ind]).long()
        self._train_neg = torch.Tensor([d[i] for i in train_neg_ind]).long()

        # 初始化用户嵌入层
        self.dim = dim
        self.items_emb_grad = None  # 预留给项目嵌入梯度的存储
        self._user_emb = nn.Embedding(1, dim)  # 单用户嵌入层
        nn.init.normal_(self._user_emb.weight, std=0.01)  # 参数初始化


    def forward(self, items_emb):
        """计算用户嵌入向量与项目嵌入向量的点积得分

        Args:
            items_emb (torch.Tensor): 项目嵌入向量组成的张量，
                形状应为(batch_size, embedding_dim)

        Returns:
            torch.Tensor: 用户与项目的匹配得分张量，
                形状为(batch_size,)，每个元素表示对应的用户-项目匹配度
        """
        # 计算用户嵌入矩阵与项目嵌入向量的逐元素乘积之和
        # 等价于用户向量和每个项目向量的点积计算
        scores = torch.sum(self._user_emb.weight * items_emb, dim=-1)
        return scores


    def train_(self, items_emb, reg=0.1):
        """训练模型参数（用户嵌入和物品嵌入）

        Args:
            items_emb (Tensor): 所有物品的初始嵌入表示
            reg (float, optional): L2正则化系数，默认为0.1

        Returns:
            tuple: 包含三个元素的元组
                - train_all (Tensor): 训练集中所有物品的索引
                - items_emb_grad (Tensor): 物品嵌入的梯度值
                - float: 当前训练步的损失值
        """
        # 准备训练所需的物品嵌入，启用梯度追踪
        items_emb = items_emb[self.train_all].clone().detach().requires_grad_(True)
        # 清空用户嵌入的梯度缓存
        self._user_emb.zero_grad()

        # 获取正负样本的嵌入表示
        pos_items_emb = items_emb[self._train_pos]
        neg_items_emb = items_emb[self._train_neg]

        # 计算正负样本的预测得分
        pos_scores = self.forward(pos_items_emb)
        neg_scores = self.forward(neg_items_emb)

        # 计算BPR损失（包含对数损失项和L2正则项）
        loss = -(pos_scores - neg_scores).sigmoid().log().sum() + \
            0.5 * (self._user_emb.weight.norm(2).pow(2) + items_emb.norm(2).pow(2)) * reg

        # 反向传播计算梯度
        loss.backward()

        # 更新用户嵌入参数（梯度下降）
        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)

        # 保存物品嵌入的梯度用于后续更新
        self.items_emb_grad = items_emb.grad

        return self.train_all, self.items_emb_grad, loss.cpu().item()


    def eval_(self, items_emb):
        """模型评估函数，计算推荐系统的各项评价指标

        Args:
            items_emb: 项目嵌入向量，作为模型前向传播的输入

        Returns:
            test_result: 测试集上的召回率指标结果数组[HR@10]
            target_result: 目标集上的综合评价指标数组[ER@5, ER@10, NDCG@10]
                当不存在目标集时返回None
        """
        # 前向传播获取项目评分
        rating = self.forward(items_emb)

        # 构建包含1个正样本和99个负样本的候选集
        items = [self._test_[0], ]  # 添加测试集正样本
        for _ in range(99):
            # 生成不在训练集且未被选中的负样本
            neg_item = np.random.randint(self.m_item)
            while neg_item in self._train_ or neg_item in items:
                neg_item = np.random.randint(self.m_item)
            items.append(neg_item)
        # 转换候选集为tensor格式
        items = torch.Tensor(items).long().to(args.device)

        # 计算采样样本的召回率指标
        sampled_hr_at_10 = evaluate_recall(rating[items], [0], 10)
        test_result = np.array([sampled_hr_at_10])

        # 计算目标集上的综合指标
        if self._target_:
            # 屏蔽训练集和测试集项目
            rating[self._train] = - (1 << 10)
            rating[self._test] = - (1 << 10)

            # 计算精确召回率和NDCG指标
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            ndcg_at_10 = evaluate_ndcg(rating, self._target_, 10)
            target_result = np.array([er_at_5, er_at_10, ndcg_at_10])
        else:
            target_result = None

        return test_result, target_result

