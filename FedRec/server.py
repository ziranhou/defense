import numpy as np
import torch
import torch.nn as nn
from mpmath import sigmoid

from parse import args

from collections import defaultdict  # 新增导入语句



class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, clients, args):
        """Initialize the embedding layer for items

        Args:
            m_item (int): Total number of unique items. Determines the size of the embedding vocabulary.
            dim (int): Dimension of the dense embeddings. Defines the size of each embedding vector.
        """
        # 父类初始化必须最先执行
        super().__init__()

        self.clients = clients  # 通过构造函数注入客户端列表
        self.args = args  # 存储参数对象
        self.client_trust = {}  # {client_id: trust_score}
        self.abnormal_reports = []  # 存储异常报告
        self.voting_records = defaultdict(list)  # {target_id: [reporter_ids]}
        # 保存基础参数用于后续操作
        self.m_item = m_item  # 物品总数
        self.dim = dim        # 嵌入向量维度

        # 创建可训练的物品嵌入矩阵 [m_item x dim]
        # 每个物品ID会被映射为dim维的稠密向量
        self.items_emb = nn.Embedding(m_item, dim)

        # 对嵌入层权重进行正态分布初始化（重要初始化策略）
        # 标准差设为0.01以防止梯度爆炸/消失
        nn.init.normal_(self.items_emb.weight, std=0.01)

    def calculate_trust_score(self, client_id):
        """动态信任度计算"""
        base_score = 0.5
        # 1. 历史行为因子
        history_factor = self.client_trust.get(client_id, 1.0)
        # 2. 异常报告因子
        report_count = len([r for r in self.abnormal_reports if r['target'] == client_id])
        report_factor = np.exp(-report_count * 0.1)
        # 3. 投票验证因子
        vote_ratio = len(set(self.voting_records[client_id])) / len(self.clients)
        # server.py中calculate_trust_score方法
        vote_factor = 1 - sigmoid(vote_ratio - self.args.voting_ratio)  # 使用self.args

        return base_score * history_factor * report_factor * vote_factor

    def process_reports(self):
        """处理异常报告并更新信任度"""
        for report in self.abnormal_reports:
            target = report['target']
            # 投票统计
            self.voting_records[target].append(report['reporter'])
            # 信任度衰减
            self.client_trust[target] *= 0.9

        # 触发处置的条件
        for target, reporters in self.voting_records.items():
            # server.py中process_reports方法
            if len(reporters) / len(self.clients) > self.args.voting_ratio:  # 使用self.args
                print(f"[Defense] 隔离客户端 {target}")
                self.client_trust[target] = 0.0

    def robust_aggregate(self, grads, clients, weights=None):
        """鲁棒聚合增强版"""
        # 1. 基于信任度的加权平均

        weights = torch.softmax(torch.tensor(weights), dim=0)
        weighted_grad = torch.sum(grads * weights.view(-1, 1, 1), dim=0)

        # 2. 异常值过滤（改进的Trimmed Mean）
        sorted_grad, _ = torch.sort(grads, dim=0)
        trim_size = int(len(clients) * 0.2)
        trimmed_mean = torch.mean(sorted_grad[trim_size:-trim_size], dim=0)

        # 3. 最终梯度融合
        return args.trust_threshold * weighted_grad + (1 - args.trust_threshold) * trimmed_mean

    def train_(self, clients, batch_clients_idx):
        """执行一批客户端的训练过程，聚合梯度并更新项目嵌入权重

        Args:
            clients: list[Client] 客户端对象列表
            batch_clients_idx: list[int] 本批次参与训练的客户端索引列表

        Returns:
            list[float]: 本批次各客户端的训练损失值集合（可能包含None值）
        """

        self.clients = clients  # 每次训练时更新客户端列表

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

