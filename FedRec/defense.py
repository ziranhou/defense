import numpy as np
import torch


class DynamicDefense:
    @staticmethod
    def adjust_threshold(server, epoch):
        """动态调整阈值规则"""
        # 根据历史异常比例调整

        if len(server.clients) == 0:  # 新增空列表保护
            return  # 或设置默认阈值

        abnormal_rate = len(server.abnormal_reports) / len(server.clients)
        if abnormal_rate > 0.2:
            server.args.trust_threshold = min(1.0, server.args.trust_threshold + 0.05)
        else:
            server.args.trust_threshold = max(0.5, server.args.trust_threshold - 0.03)
    @staticmethod
    def broadcast_rules(server, clients):
        """规则同步（增加防御性校验）"""
        new_rules = {
            'threshold': server.args.trust_threshold,
            'penalty': 0.9 if len(server.abnormal_reports) > 10 else 0.95,
            'noise_std': 1e-6  # 新增噪声参数
        }

        # 添加客户端类型校验
        for client in clients:
            if hasattr(client, 'update_rules'):
                client.update_rules(new_rules)
            else:
                print(f"警告：客户端{type(client)}不支持规则更新")
            client.update_rules(new_rules)

    def adjust_threshold(server, epoch):
        if len(server.clients) == 0:
            return

        # 引入滑动窗口机制
        window_size = min(10, epoch)
        recent_reports = server.abnormal_reports[-window_size:]
        abnormal_rate = len(recent_reports) / (window_size * len(server.clients))

        # 自适应调整步长
        step = 0.1 * (2 / (1 + np.exp(-abnormal_rate * 10)) - 1)

        server.args.trust_threshold = np.clip(
            server.args.trust_threshold + step,
            0.5, 0.9
        )

    class EnhancedDefense:
        @staticmethod
        def gradient_validation(grads):
            """梯度指纹校验"""
            # 计算梯度分布统计量
            mean = torch.mean(grads, dim=0)
            std = torch.std(grads, dim=0)
            # 检测3σ异常
            z_scores = (grads - mean) / std
            return torch.any(torch.abs(z_scores) > 3, dim=1)

        @staticmethod
        def temporal_consistency_check(client):
            """时序行为一致性检测"""
            # 计算最近5次更新差异
            diffs = [torch.norm(client.history[i] - client.history[i - 1])
                     for i in range(1, len(client.history))]
            return np.std(diffs) > 0.1 * np.mean(diffs)

    # FedRec/defense.py 新增内容

    def defense_evaluation(server, attack_clients, benign_clients, clean_grad):
        """防御效果评估函数（需在文件顶部导入：from sklearn.metrics.pairwise import cosine_similarity）"""
        # 计算关键指标
        metrics = {
            '隔离准确率': len(server.blocklist & attack_clients) / len(attack_clients),
            '误隔离率': len(server.blocklist - attack_clients) / len(benign_clients),
            '平均信任度': np.mean(list(server.client_trust.values())),
            '梯度相似度': cosine_similarity(clean_grad, server.grad)
        }
        return metrics
