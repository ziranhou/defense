import numpy as np
import torch
import torch.nn as nn
from parse import args


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim):
        """Initialize the embedding layer for items

        Args:
            m_item (int): Total number of unique items. Determines the size of the embedding vocabulary.
            dim (int): Dimension of the dense embeddings. Defines the size of each embedding vector.
        """
        # 父类初始化必须最先执行
        super().__init__()

        # 保存基础参数用于后续操作
        self.m_item = m_item  # 物品总数
        self.dim = dim        # 嵌入向量维度

        # 创建可训练的物品嵌入矩阵 [m_item x dim]
        # 每个物品ID会被映射为dim维的稠密向量
        self.items_emb = nn.Embedding(m_item, dim)

        # 对嵌入层权重进行正态分布初始化（重要初始化策略）
        # 标准差设为0.01以防止梯度爆炸/消失
        nn.init.normal_(self.items_emb.weight, std=0.01)


    def train_(self, clients, batch_clients_idx):
        """执行一批客户端的训练过程，聚合梯度并更新项目嵌入权重

        Args:
            clients: list[Client] 客户端对象列表
            batch_clients_idx: list[int] 本批次参与训练的客户端索引列表

        Returns:
            list[float]: 本批次各客户端的训练损失值集合（可能包含None值）
        """
        # 初始化本批次梯度累积和损失收集容器
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(self.items_emb.weight)

        # 遍历本批次选择的客户端进行训练
        for idx in batch_clients_idx:
            client = clients[idx]
            # 获取客户端训练产生的梯度信息（项目索引、梯度、损失值）
            items, items_emb_grad, loss = client.train_(self.items_emb.weight)

            # 梯度裁剪与聚合处理
            with torch.no_grad():
                # 计算梯度L2范数并识别超出阈值的梯度
                norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                too_large = norm[:, 0] > args.grad_limit
                # 对过大的梯度进行缩放处理
                items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit)
                # 将处理后的梯度累加到批次梯度矩阵
                batch_items_emb_grad[items] += items_emb_grad

            # 收集有效的损失值
            if loss is not None:
                batch_loss.append(loss)

        # 执行参数更新（梯度下降）
        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
        return batch_loss


    def eval_(self, clients):
        """
        在给定客户端集合上执行评估，计算平均测试结果和目标任务结果

        Args:
            clients (list): 客户端对象列表，每个客户端需实现eval_方法

        Returns:
            tuple: 包含两个元素的元组，分别表示：
                - 所有客户端测试结果的平均值（float）
                - 所有客户端目标任务结果的平均值（float）
        """
        # 初始化计数器和结果累加器
        test_cnt = 0
        test_results = 0.
        target_cnt = 0
        target_results = 0.

        # 遍历所有客户端进行结果收集
        for client in clients:
            # 获取单个客户端的评估结果（可能包含None值）
            test_result, target_result = client.eval_(self.items_emb.weight)

            # 处理有效的测试结果
            if test_result is not None:
                test_cnt += 1
                test_results += test_result

            # 处理有效的目标任务结果
            if target_result is not None:
                target_cnt += 1
                target_results += target_result

        # 计算并返回两个维度的平均结果
        return test_results / test_cnt, target_results / target_cnt


# server.py
import torch
import torch.nn as nn
from parse import args


def weighted_median(client_updates, client_losses):
    """
    计算加权中位数聚合。
    通过加权聚合每个客户端的更新，使用损失值作为权重。
    """
    # 将每个客户端的更新按照其损失加权
    sorted_updates = sorted(zip(client_losses, client_updates), key=lambda x: x[0])
    sorted_client_losses = [s[0] for s in sorted_updates]
    sorted_client_updates = [s[1] for s in sorted_updates]

    # 计算加权中位数
    total_loss = sum(sorted_client_losses)
    half_loss = total_loss / 2
    cumulative_loss = 0
    for loss, update in zip(sorted_client_losses, sorted_client_updates):
        cumulative_loss += loss
        if cumulative_loss >= half_loss:
            return update
    return sorted_client_updates[0]  # If not found, return the first update


class FedRecServerWithDefense(nn.Module):
    def __init__(self, m_item, dim, grad_limit=1.0, clients_limit=0.05):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.grad_limit = grad_limit
        self.clients_limit = clients_limit
        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)


    def train_(self, clients, batch_clients_idx):
        batch_loss = []
        all_losses = []
        client_updates = []

        for idx in batch_clients_idx:
            client = clients[idx]
            items, items_emb_grad, loss = client.train_(self.items_emb.weight)

            # 记录损失
            all_losses.append(loss)

            # -- 关键：将 [len(items), dim] 大小的梯度扩展到 [m_item, dim] --
            full_update = torch.zeros_like(self.items_emb.weight)  # [m_item, dim]
            full_update[items] = items_emb_grad  # 把对应索引的更新写入full_update
            client_updates.append(full_update)

        # ----过滤 None，避免加权中位数时出错----
        valid_pairs = [
            (update, loss) for update, loss in zip(client_updates, all_losses) if loss is not None
        ]
        if not valid_pairs:
            # 若本批次全是None损失，就不更新
            return all_losses

        # 解包得到过滤后的张量和loss
        filtered_updates, filtered_losses = zip(*valid_pairs)

        # 使用加权中位数聚合
        robust_update = weighted_median(filtered_updates, filtered_losses)
        # robust_update 现在是 [m_item, dim] 大小，能与 self.items_emb.weight 匹配

        # 更新全局物品嵌入
        with torch.no_grad():
            self.items_emb.weight.data.add_(robust_update, alpha=-args.lr)

        return all_losses
    def eval_(self, clients):
        """
        Evaluate the model on the clients.
        """
        test_cnt = 0
        test_results = 0.
        target_cnt = 0
        target_results = 0.

        for client in clients:
            test_result, target_result = client.eval_(self.items_emb.weight)
            if test_result is not None:
                test_cnt += 1
                test_results += test_result
            if target_result is not None:
                target_cnt += 1
                target_results += target_result

        return test_results / test_cnt if test_cnt > 0 else 0, target_results / target_cnt if target_cnt > 0 else 0

# 0011

