import argparse
import torch.cuda as cuda


def parse_args():
    """
    解析命令行参数并返回参数对象，用于推荐系统模型的运行和攻击配置

    Returns:
        argparse.Namespace: 包含所有解析后参数的命名空间对象

    参数说明:
        --attack (str): 指定使用的攻击方法名称，默认为'FedRecAttack'
        --dim (int): 潜在向量维度大小，默认为32
        --path (str): 输入数据存储路径，默认为'Data/'
        --dataset (str): 选择使用的数据集名称
        --device (str): 指定运行设备(cuda/cpu)，自动检测GPU可用性，默认优先使用cuda

        --lr (float): 学习率参数，默认为0.01
        --epochs (int): 训练的总轮次数，默认为200
        --batch_size (int): 训练批大小，默认为256

        --grad_limit (float): 项目梯度L2范数的阈值限制，默认为1.0
        --clients_limit (float): 恶意客户端比例上限，默认允许5%
        --items_limit (int): 非零项目梯度的数量限制，默认最多60个
        --part_percent (int): 攻击者先验知识占比，默认1%

        --attack_lr (float): FedRecAttack专用学习率，默认为0.01
        --attack_batch_size (int): FedRecAttack批大小，默认为256
    """
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    # 基础模型配置参数
    parser.add_argument('--attack', nargs='?', default='FedRecAttack', help="Specify a attack method")
    parser.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cuda' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    # 训练参数配置
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')

    # 防御机制限制参数
    parser.add_argument('--grad_limit', type=float, default=1., help='Limit of l2-norm of item gradients.')
    parser.add_argument('--clients_limit', type=float, default=0.05, help='Limit of proportion of malicious clients.')
    parser.add_argument('--items_limit', type=int, default=60, help='Limit of number of non-zero item gradients.')
    parser.add_argument('--part_percent', type=int, default=1, help='Proportion of attacker\'s prior knowledge.')

    # 攻击参数配置
    parser.add_argument('--attack_lr', type=float, default=0.01, help='Learning rate on FedRecAttack.')
    parser.add_argument('--attack_batch_size', type=int, default=256, help='Batch size on FedRecAttack.')




    return parser.parse_args()



args = parse_args()
