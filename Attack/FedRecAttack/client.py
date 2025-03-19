import torch
import numpy as np
from parse import args


class FedRecAttackClient(object):
    def __init__(self, center, train_all_size):
        """初始化类的实例

        参数说明：
        center: 中心点坐标/位置信息，用于表示当前实例的核心定位基准
        train_all_size: 训练数据总容量，表示完整训练数据集的大小或数量

        属性说明：
        self._center: (受保护属性)存储实例的中心点配置
        self._train_all_size: (受保护属性)记录训练集的完整规模参数
        self.train_all: 用于存储完整训练数据的容器，初始状态为未加载
        """
        self._center = center
        self._train_all_size = train_all_size
        self.train_all = None


    def eval_(self, _items_emb):
        """
        评估物品嵌入向量的效果（函数尚未实现）

        Args:
            _items_emb (torch.Tensor|np.ndarray): 物品的特征嵌入向量/张量，形状通常为 [num_items, embedding_dim]

        Returns:
            (None, None): 返回两个空值占位符。按常规实现预期应返回：
                - 第一个None可能表示预测结果张量
                - 第二个None可能表示真实标签张量
        """
        return None, None


    @staticmethod
    def noise(shape, std):
        """
        生成符合多维正态分布的噪声张量，并将其移动到指定设备。

        参数:
            shape (tuple): 期望输出噪声张量的维度，通常为(batch_size, feature_dim)
            std (float): 每个维度的标准差（协方差矩阵对角线值为std的平方，当前实现存在标准差/方差参数混淆问题）

        返回:
            torch.Tensor: 生成的噪声张量，形状为(shape[0], shape[1])，位于args.device指定的设备
        """
        # 生成多维正态分布噪声，注意当前协方差矩阵实际使用std而非std²
        # 这将导致实际标准差为sqrt(std)而非std（需确认设计意图）
        noise = np.random.multivariate_normal(
            mean=np.zeros(shape[1]), cov=np.eye(shape[1]) * std, size=shape[0]
        )

        # 将numpy数组转换为PyTorch张量并迁移到目标设备
        return torch.Tensor(noise).to(args.device)


    def train_(self, items_emb, std=1e-7):
        """训练攻击模型并处理梯度

        Args:
            items_emb: 物品嵌入矩阵，表示当前批次的物品特征向量
            std: 噪声标准差，控制添加到梯度中的随机噪声强度，默认1e-7

        Returns:
            tuple: 包含三个元素的元组
                - train_all: 经过筛选的训练样本索引集合
                - items_emb_grad: 处理后的物品嵌入梯度矩阵
                - None: 占位符
        """
        # 中心模型训练阶段
        self._center.train_(items_emb, args.attack_batch_size)

        with torch.no_grad():
            # 获取中心模型的梯度信息
            items_emb_grad = self._center.items_emb_grad

            # 初始化训练样本选择策略
            if self.train_all is None:
                # 获取目标物品索引并转换为设备张量
                target_items = self._center.target_items
                target_items_ = torch.Tensor(target_items).long().to(args.device)

                # 计算带噪声的梯度范数作为采样概率
                p = (items_emb_grad + self.noise(items_emb_grad.shape, std)).norm(2, dim=-1)
                p[target_items_] = 0.  # 排除目标物品的采样概率
                p = (p / p.sum()).cpu().numpy()  # 归一化为概率分布

                # 组合目标物品和随机采样物品
                rand_items = np.random.choice(
                    np.arange(len(p)), self._train_all_size - len(target_items), replace=False, p=p
                ).tolist()
                self.train_all = torch.Tensor(target_items + rand_items).long().to(args.device)

            # 梯度裁剪与噪声注入
            items_emb_grad = items_emb_grad[self.train_all]
            items_emb_grad_norm = items_emb_grad.norm(2, dim=-1, keepdim=True)

            # 梯度限幅处理：限制最大梯度范数
            grad_max = args.grad_limit
            too_large = items_emb_grad_norm[:, 0] > grad_max
            items_emb_grad[too_large] /= (items_emb_grad_norm[too_large] / grad_max)

            # 添加随机噪声增强鲁棒性
            items_emb_grad += self.noise(items_emb_grad.shape, std)

            # 更新中心模型的梯度信息
            self._center.items_emb_grad[self.train_all] -= items_emb_grad
        return self.train_all, items_emb_grad, None

