import torch
import numpy as np
import torch.nn as nn
from parse import args


class FedRecAttackCenter(nn.Module):
    def __init__(self, users_train_ind, target_items, m_item, dim):
        """初始化推荐系统模型组件

        Args:
            users_train_ind (array-like): 训练用户的索引列表/数组，用于标识参与训练的用户
            target_items (array-like): 目标物品的索引列表/数组，表示需要预测的物品集合
            m_item (int): 物品总量，表示系统中所有物品的数量
            dim (int): 嵌入维度，控制用户和物品向量的维度大小

        Returns:
            None: 构造函数不返回任何值
        """
        # 调用父类构造函数完成基础初始化
        super().__init__()

        # 参数存储模块：保存输入参数并计算派生参数
        self.users_train_ind = users_train_ind
        self.target_items = target_items
        self.n_user = len(users_train_ind)  # 计算参与训练的用户总数
        self.m_item = m_item
        self.dim = dim

        # 组件初始化占位模块：后续将被替换为实际参数的占位符
        self._opt = None        # 优化器占位
        self.items_emb = None   # 物品嵌入矩阵占位
        self.users_emb = None   # 用户嵌入矩阵占位
        self.items_emb_grad = None  # 物品嵌入梯度占位

        # 用户嵌入层初始化：使用正态分布初始化可训练的用户嵌入矩阵
        self.users_emb_ = nn.Embedding(self.n_user, dim)
        nn.init.normal_(self.users_emb_.weight, std=0.01)  # 标准差0.01的正态分布初始化


    @property
    def opt(self):
        """获取攻击模型的优化器(延迟初始化)

        属性说明:
            本属性使用惰性初始化策略，在首次访问时创建优化器实例。
            后续访问直接返回已创建的优化器，避免重复初始化

        返回:
            torch.optim.Adam: 基于用户嵌入权重的Adam优化器实例，
                学习率从全局args对象中获取attack_lr参数
        """
        # 延迟初始化逻辑：仅在首次访问时创建优化器
        if self._opt is None:
            # 初始化Adam优化器，仅优化用户嵌入层的权重参数
            # 学习率使用攻击专用的超参数attack_lr
            self._opt = torch.optim.Adam([self.users_emb_.weight], lr=args.attack_lr)
        return self._opt


    def loss(self, batch_users):
        """计算基于BPR（Bayesian Personalized Ranking）的损失函数

        参数:
            batch_users (list[int]): 当前批次的用户ID列表，每个用户ID将与其正样本物品生成配对数据

        返回值:
            torch.Tensor: 计算得到的损失值标量，形状为(1,)

        功能说明:
        1. 为每个用户生成正负样本对
        2. 使用矩阵分解计算用户和物品的嵌入向量
        3. 通过对比正负样本对的得分计算排序损失
        """
        # 初始化三元组容器：用户ID，正样本物品ID，负样本物品ID
        users, pos_items, neg_items = [], [], []

        # 生成正负样本对
        for user in batch_users:
            # 遍历该用户所有训练集中的正样本物品
            for pos_item in self.users_train_ind[user]:
                # 添加用户-正样本基础对
                users.append(user)
                pos_items.append(pos_item)

                # 生成有效负样本（不在用户的正样本集合中）
                neg_item = np.random.randint(self.m_item)
                while neg_item in self.users_train_ind[user]:
                    neg_item = np.random.randint(self.m_item)
                neg_items.append(neg_item)

        # 将数据转换为GPU张量（如果可用）
        users = torch.Tensor(users).long().to(args.device)
        pos_items = torch.Tensor(pos_items).long().to(args.device)
        neg_items = torch.Tensor(neg_items).long().to(args.device)

        # 获取用户/物品的嵌入向量
        users_emb = self.users_emb_(users)          # 形状: (batch_size, emb_dim)
        pos_items_emb = self.items_emb[pos_items]   # 形状: (batch_size, emb_dim)
        neg_items_emb = self.items_emb[neg_items]   # 形状: (batch_size, emb_dim)

        # 计算正负样本对的得分
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)  # 形状: (batch_size,)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)  # 形状: (batch_size,)

        # 计算BPR损失函数（对数sigmoid损失）
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()
        return loss


    def train_(self, new_items_emb, batch_size, eps=1e-4):
        """训练模型参数并更新用户/物品嵌入

        Args:
            new_items_emb (Tensor): 新物品嵌入张量，用于更新当前物品嵌入
            batch_size (int): 训练时用户批处理的大小
            eps (float, optional): 判断物品嵌入是否变化的阈值，默认1e-4

        Returns:
            None: 直接修改类内部状态，无显式返回值
        """
        # 防御性拷贝并切断计算图，确保原始数据不被修改
        new_items_emb = new_items_emb.clone().detach()
        # 物品嵌入未发生显著变化时提前终止训练
        if (self.items_emb is not None) and ((new_items_emb - self.items_emb).abs().sum() < eps):
            return
        self.items_emb = new_items_emb

        # 用户嵌入训练阶段
        if len(self.users_train_ind):
            # 随机洗牌用户顺序进行批训练
            rand_users = np.arange(self.n_user)
            np.random.shuffle(rand_users)
            total_loss = 0.
            # 分批次训练并更新参数
            for i in range(0, len(rand_users), batch_size):
                loss = self.loss(rand_users[i: i + batch_size])
                total_loss += loss.cpu().item()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            total_loss /= len(rand_users)
        else:
            # 无训练数据时初始化用户嵌入
            nn.init.normal_(self.users_emb_.weight, std=0.1)
        self.users_emb = self.users_emb_.weight.clone().detach()

        # 构建负采样忽略列表（已交互项和目标项）
        rand_users = np.arange(self.n_user)
        ignore_users, ignore_items = [], []
        for idx, user in enumerate(rand_users):
            # 收集需要屏蔽的目标物品
            for item in self.target_items:
                if item not in self.users_train_ind[user]:
                    ignore_users.append(idx)
                    ignore_items.append(item)
            # 屏蔽已训练物品
            for item in self.users_train_ind[user]:
                ignore_users.append(idx)
                ignore_items.append(item)
        ignore_users = torch.Tensor(ignore_users).long().to(args.device)
        ignore_items = torch.Tensor(ignore_items).long().to(args.device)

        # 计算top-k推荐并屏蔽指定项
        with torch.no_grad():
            users_emb = self.users_emb[torch.Tensor(rand_users).long().to(args.device)]
            items_emb = self.items_emb
            scores = torch.matmul(users_emb, items_emb.t())
            scores[ignore_users, ignore_items] = - (1 << 10)  # 用极大负数屏蔽指定位置
            _, top_items = torch.topk(scores, 10)  # 获取每个用户top10物品
        top_items = top_items.cpu().tolist()

        # 构建三元组训练数据（用户，正样本，负样本）
        users, pos_items, neg_items = [], [], []
        for idx, user in enumerate(rand_users):
            for item in self.target_items:
                if item not in self.users_train_ind[user]:
                    users.append(user)
                    pos_items.append(item)
                    neg_items.append(top_items[idx].pop())  # 使用top列表中最后一个作为负样本
        # 转换数据为张量格式
        users = torch.Tensor(users).long().to(args.device)
        pos_items = torch.Tensor(pos_items).long().to(args.device)
        neg_items = torch.Tensor(neg_items).long().to(args.device)

        # 计算对比损失并进行反向传播
        users_emb = self.users_emb[users]
        items_emb = self.items_emb.clone().detach().requires_grad_(True)
        pos_items_emb = items_emb[pos_items]
        neg_items_emb = items_emb[neg_items]
        # 计算正负样本得分差异
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = neg_scores - pos_scores
        # 对负分差大于正分差的情况应用指数惩罚
        loss[loss < 0] = torch.exp(loss[loss < 0]) - 1
        loss = loss.sum()
        loss.backward()
        # 保存物品嵌入梯度供后续使用
        self.items_emb_grad = items_emb.grad

