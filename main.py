import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset

from FedRec.server import FedRecServer
from FedRec.client import FedRecClient

from FedRec.server import FedRecServerWithDefense



def setup_seed(seed):
    """
    初始化随机数生成器的种子，用于实验结果的确定性复现

    本函数通过设置多个随机数生成器后端的种子值，确保在CPU、GPU环境下的
    随机操作可重复。适用于PyTorch、NumPy和Python内置random模块。

    Args:
        seed (int): 全局随机种子值，建议使用大于0的整数。
                   该值将应用于所有相关的随机数生成器

    Returns:
        None: 本函数没有返回值
    """

    # 设置PyTorch相关种子（包含CPU和CUDA设备）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU场景下的额外设置

    # 设置NumPy随机种子
    np.random.seed(seed)

    # 设置Python内置随机模块种子
    random.seed(seed)

    # 启用CuDNN的确定性算法模式
    # 警告：可能降低部分模型的训练速度，但能确保可重复性
    torch.backends.cudnn.deterministic = True



def main():
    """联邦推荐系统主函数，负责整体流程控制

    功能:
        1. 加载并预处理数据集
        2. 初始化服务器和客户端
        3. 根据攻击类型配置恶意客户端
        4. 执行模型训练与评估流程
        5. 输出训练过程中的性能指标

    流程说明:
        - 参数解析与日志输出
        - 数据集加载与目标物品采样
        - 服务器/客户端初始化
        - 恶意客户端注入配置
        - 模型初始化性能评估
        - 多轮次训练循环
        - 性能结果收集与展示

    注意:
        通过args参数对象获取所有配置参数，包含:
        - path: 数据集路径
        - dataset: 数据集名称
        - device: 计算设备(cpu/cuda)
        - dim: 嵌入维度
        - clients_limit: 恶意客户端比例
        - attack: 攻击类型(FedRecAttack/Random/Popular/Bandwagon)
        - items_limit: 项目数量限制
        - epochs: 训练轮次
        - batch_size: 批量大小
    """
    # 参数格式化输出
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    # 初始化计时器
    t0 = time()

    # 加载数据集并采样目标物品
    m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity = load_dataset(args.path + args.dataset)
    target_items = np.random.choice(m_item, 1, replace=False).tolist()

    # 初始化服务器和基础客户端
    server = FedRecServerWithDefense(m_item, args.dim).to(args.device)
    # server = FedRecServer(m_item, args.dim).to(args.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        clients.append(
            FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
        )

    # 配置恶意客户端攻击模块
    malicious_clients_limit = int(len(clients) * args.clients_limit)
    if args.attack == 'FedRecAttack':
        # 联邦推荐系统专用攻击模式
        from Attack.FedRecAttack.center import FedRecAttackCenter
        from Attack.FedRecAttack.client import FedRecAttackClient

        attack_center = FedRecAttackCenter(part_train_ind, target_items, m_item, args.dim).to(args.device)
        for _ in range(malicious_clients_limit):
            clients.append(FedRecAttackClient(attack_center, args.items_limit))

    else:
        # 基准攻击模式处理
        from Attack.baseline import BaselineAttackClient

        if args.attack == 'Random':
            # 随机攻击：随机选择非目标物品填充
            for _ in range(malicious_clients_limit):
                train_ind = [i for i in target_items]
                for __ in range(args.items_limit // 2 - len(target_items)):
                    item = np.random.randint(m_item)
                    while item in train_ind:
                        item = np.random.randint(m_item)
                    train_ind.append(item)
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        elif args.attack == 'Popular':
            # 流行物品攻击：选择最流行的物品进行注入
            for i in target_items:
                items_popularity[i] = 1e10
            _, train_ind = torch.Tensor(items_popularity).topk(args.items_limit // 2)
            train_ind = train_ind.numpy().tolist()
            for _ in range(malicious_clients_limit):
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        elif args.attack == 'Bandwagon':
            # 流行混合攻击：结合目标物品和部分流行物品
            for i in target_items:
                items_popularity[i] = - 1e10
            items_limit = args.items_limit // 2
            _, popular_items = torch.Tensor(items_popularity).topk(m_item // 10)
            popular_items = popular_items.numpy().tolist()

            for _ in range(malicious_clients_limit):
                train_ind = [i for i in target_items]
                train_ind += np.random.choice(popular_items, int(items_limit * 0.1), replace=False).tolist()
                rest_items = []
                for i in range(m_item):
                    if i not in train_ind:
                        rest_items.append(i)
                train_ind += np.random.choice(rest_items, items_limit - len(train_ind), replace=False).tolist()
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        else:
            print('Unknown args --attack.')
            return

    # 输出数据集统计信息
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({Sampled HR@10}), ({ER@5},{ER@10},{NDCG@10})")

    # 初始模型性能评估
    t1 = time()
    with torch.no_grad():
        test_result, target_result = server.eval_(clients)
    print("Iteration 0(init), (%.4f) on test" % tuple(test_result) +
          ", (%.4f, %.4f, %.4f) on target." % tuple(target_result) +
          " [%.1fs]" % (time() - t1))

    # 主训练循环
    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            # 分批次训练
            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx)

                # 过滤掉 None 值
                loss = [l for l in loss if l is not None]  # 过滤掉 None 值
                total_loss.extend(loss)

            # 计算总损失并防止 None 引起的错误
            if total_loss:  # 如果 total_loss 不是空的
                total_loss = np.mean(total_loss).item()
            else:
                total_loss = 0  # 如果没有有效的损失，设置损失为 0

            # 本轮次性能评估
            t2 = time()
            with torch.no_grad():
                test_result, target_result = server.eval_(clients)
            print("Iteration %d, loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1) +
                  ", (%.4f) on test" % tuple(test_result) +
                  ", (%.4f, %.4f, %.4f) on target." % tuple(target_result) +
                  " [%.1fs]" % (time() - t2))
    except KeyboardInterrupt:
        pass



setup_seed(20211111)

if __name__ == "__main__":
    main()
