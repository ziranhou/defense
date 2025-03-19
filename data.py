import math
import numpy as np
from parse import args


def load_file(file_path):
    """
    加载并解析数据文件，获取最大项目编号和所有位置列表

    参数:
        file_path (str): 待读取数据文件的路径，文件每行格式为: 前缀数字+空格分隔的位置编号

    返回:
        tuple: 包含两个元素的元组
            - m_item (int): 所有位置编号中的最大值加1（用于确定项目总数）
            - all_pos (list): 二维列表，每个子列表对应文件中的一行处理后的位置编号（排除第一个元素）
    """
    m_item, all_pos = 0, []

    # 读取文件并逐行处理
    with open(file_path, "r") as f:
        for line in f.readlines():
            # 转换每行数据为整数列表，排除第一个元素后作为位置列表
            pos = list(map(int, line.rstrip().split(' ')))[1:]

            # 更新最大项目编号（当前行最大编号+1与历史最大值比较）
            if pos:
                m_item = max(m_item, max(pos) + 1)

            # 将当前行的位置列表添加到总列表中
            all_pos.append(pos)

    return m_item, all_pos



def load_dataset(path):
    """
    加载数据集并统计项目流行度信息

    参数:
        path (str): 数据集目录路径，包含训练集和测试集文件

    返回:
        tuple: 包含以下元素的元组:
            - m_item (int): 数据集中最大的项目ID(从1开始计数)
            - all_train_ind (list): 完整训练集的交互记录，每个元素是用户的项目交互列表
            - all_test_ind (list): 测试集的交互记录，结构同all_train_ind
            - part_train_ind (list): 部分采样后的训练集(当args.part_percent>0时有效)
            - items_popularity (ndarray): 每个项目的总出现次数统计数组，索引对应项目ID

    功能说明:
        1. 加载完整训练集和测试集，确定项目ID的最大范围
        2. 根据配置可能加载部分采样的训练集
        3. 统计所有项目在训练集和测试集中的总出现次数
    """
    m_item = 0
    # 加载完整训练集和测试集，并获取最大项目ID
    m_item_, all_train_ind = load_file(path + "train.txt")
    m_item = max(m_item, m_item_)
    m_item_, all_test_ind = load_file(path + "test.txt")
    m_item = max(m_item, m_item_)

    # 根据配置加载部分采样训练集
    if args.part_percent > 0:
        _, part_train_ind = load_file(path + "train.part-{}%.txt".format(args.part_percent))
    else:
        part_train_ind = []

    # 统计项目流行度(训练集+测试集总出现次数)
    items_popularity = np.zeros(m_item)
    for items in all_train_ind:
        for item in items:
            items_popularity[item] += 1
    for items in all_test_ind:
        for item in items:
            items_popularity[item] += 1

    return m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity



def sample_part_of_dataset(path, ratio):
    """从数据集中按比例采样部分正样本数据，生成新的训练文件

    Args:
        path (str): 数据集文件所在目录路径，要求目录下包含train.txt文件
        ratio (float): 采样比例，取值范围[0-1]，表示保留原数据集中每个用户正样本的比例

    Returns:
        None: 无直接返回值，结果会写入新生成的训练文件中
    """
    # 加载原始训练数据，all_pos结构为[[user1_items], [user2_items], ...]
    _, all_pos = load_file(path + "train.txt")

    # 创建新训练文件，文件名包含采样比例百分比
    with open(path + "train.part-{}%.txt".format(int(ratio * 100)), "w") as f:
        # 遍历每个用户的全部正样本
        for user, pos_items in enumerate(all_pos):
            # 按比例随机采样不重复的样本，使用math.ceil确保至少保留1个样本
            part_pos_items = np.random.choice(pos_items, math.ceil(len(pos_items) * ratio), replace=False).tolist()

            # 写入新格式的训练数据：用户ID + 空格分隔的item列表
            f.write(str(user))
            for item in part_pos_items:
                f.write(' ' + str(item))
            f.write('\n')

