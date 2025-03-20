# FedRec/monitor.py

import matplotlib.pyplot as plt
import numpy as np


class DefenseMonitor:
    def __init__(self, save_path="./defense_monitor"):
        self.save_path = save_path

    def plot_trust_distribution(self, server):
        """可视化信任度分布（建议在Jupyter Notebook中调用）"""
        plt.figure(figsize=(10, 6))
        plt.hist(list(server.client_trust.values()),
                 bins=np.linspace(0, 1, 21),
                 alpha=0.7,
                 edgecolor='black')
        plt.axvline(server.args.trust_threshold,
                    color='r',
                    linestyle='--',
                    label=f'当前阈值({server.args.trust_threshold})')
        plt.title('客户端信任度分布')
        plt.xlabel('信任度')
        plt.ylabel('客户端数量')
        plt.legend()
        plt.savefig(f"{self.save_path}/trust_distribution.png")
        plt.close()

    def plot_attack_pattern(self, grads, epoch):
        """梯度模式热力图分析"""
        plt.figure(figsize=(12, 8))
        plt.imshow(grads.cpu().numpy(),
                   cmap='viridis',
                   aspect='auto',
                   interpolation='nearest')
        plt.colorbar(label='梯度值')
        plt.title(f'梯度模式分析（Epoch {epoch}）')
        plt.xlabel('嵌入维度')
        plt.ylabel('客户端索引')
        plt.savefig(f"{self.save_path}/grad_pattern_epoch{epoch}.png")
        plt.close()
