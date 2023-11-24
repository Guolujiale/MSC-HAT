import os
import pandas as pd
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict
import numpy as np
import torch
from random import sample


# 1. 加入边的生成函数
def generate_edges(xy_coordinates, k, unique_regions):

    edges = defaultdict(list)
    # 对于同一个region的细胞构建st-st与rgb-rgb
    region_to_cell_map = defaultdict(list)
    for i, region in enumerate(unique_regions):
        region_to_cell_map[region].append(i)

    for region, cells in region_to_cell_map.items():
        local_coords = xy_coordinates[cells]
        distance_matrix = pairwise_distances(local_coords)

        for i in range(distance_matrix.shape[0]):
            knn_indices = np.argsort(distance_matrix[i])[1:k + 1]
            for index in knn_indices:
                global_i = cells[i]
                global_index = cells[index]
                edges['st_st'].append((global_i, global_index))
        print(f"Calculating cell region '{region}'")
    print("Calculating st-st edge done")
    return edges

# 2. 加入数据加载和划分函数
def load_data(filename, sample_rate=1):
    df = pd.read_csv(filename,encoding='ISO-8859-1')
    df = df.sample(n=round(sample_rate * len(df)), random_state=1)
    return df


def generate_edge_label_matrix(edges, num_nodes, edge_type='st_st', multiplier=5):
    """
    生成指定类型边的边标签矩阵。
    :param edges: 包含所有边的字典。
    :param num_nodes: 图中的节点数。
    :param edge_type: 要处理的边类型。
    :param multiplier: 要生成的负样本数量与正样本数量的比例。
    :return: 边标签矩阵，其中每一行包含两个节点索引和一个标签（1表示有边，0表示无边）。
    """
    # 获取正样本边
    positive_edges = set(edges[edge_type])
    # 计算需要的负样本数量
    num_negative_samples = len(positive_edges) * multiplier
    # 生成负样本
    negative_edges = set()
    while len(negative_edges) < num_negative_samples:
        # 随机选择节点对
        sampled_edges = {(np.random.randint(num_nodes), np.random.randint(num_nodes)) for _ in
                         range(num_negative_samples)}

        # 删除已存在的边
        sampled_edges = sampled_edges - positive_edges
        # 添加到负样本集合
        negative_edges.update(sampled_edges)
        # 如果负样本过多，则进行截断
        if len(negative_edges) > num_negative_samples:
            negative_edges = set(sample(negative_edges, num_negative_samples))
    # 合并正负样本并创建标签
    all_edges = [(edge[0], edge[1], 1) for edge in positive_edges] + \
                [(edge[0], edge[1], 0) for edge in negative_edges]
    print("Calculating edge_label_matrix done")
    # 转换为张量
    edge_label_matrix = torch.tensor(all_edges, dtype=torch.long)

    return edge_label_matrix

# 3. 使用InMemoryDataset的方式构建图
class HeteroGraphDataset(InMemoryDataset):
    def __init__(self, filename, k=5, transform=None, output_dir="./data/unsup"):
        self.root = '.'
        super(HeteroGraphDataset, self).__init__(self.root, transform)

        if os.path.exists(output_dir):
            print("graph data already exist, loading from:", output_dir)
            self.data = torch.load(os.path.join(output_dir, "data.pt"))
            self.edge_index_list = torch.load(os.path.join(output_dir, "edge_index_list.pt"))
            self.st_indices = torch.load(os.path.join(output_dir, "st_indices.pt"))
            self.decoder_edge_label_matrix = torch.load(os.path.join(output_dir, "decoder_edge_label_matrix.pt"))
        else:
            print("there is no graph data, calculating ...")
            df = load_data(filename)
            self.data, self.edge_index_list, self.decoder_edge_label_matrix, self.st_indices = self.process_data(df, k)
            self.save_processed_data(output_dir)

    def save_processed_data(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.data, os.path.join(output_dir, "data.pt"))
        torch.save(self.edge_index_list, os.path.join(output_dir, "edge_index_list.pt"))
        torch.save(self.st_indices, os.path.join(output_dir, "st_indices.pt"))
        torch.save(self.decoder_edge_label_matrix, os.path.join(output_dir, "decoder_edge_label_matrix.pt"))

    def process_data(self, df, k):
        st_features = df.iloc[:, 0:2000].values #结束索引 49 是不包括在内的，所以实际上到第48列。
        xy_coordinates = df.iloc[:, 2000:2001].values
        unique_region = df['region'].values
        # 创建节点特征存储
        st_node_features = torch.tensor(st_features, dtype=torch.float)

        # 替换原先的knn边生成方式
        edges = generate_edges(xy_coordinates, k, unique_region)
        # 计算总的节点数量
        num_nodes = len(st_node_features)
        print(f"Total num_nodes: {num_nodes}")

        data = Data(num_nodes=num_nodes)
        max_dim = st_features.shape[1]
        # 将所有的节点特征复制到inputs_list
        inputs = torch.zeros((num_nodes, max_dim))
        inputs[:len(st_node_features), :st_features.shape[1]] = st_node_features

        # 创建空的edge_index列表
        edge_index_list = []
        # 对于每种边的关系，创建一个偏置矩阵并添加到列表中
        for edge_type, edge_list in edges.items():
            # 创建一个空的偏置矩阵
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_index_list.append(edge_index)
            print("calculate edge_index done")
            setattr(data, f'{edge_type}_index', edge_index)

        # 将edges也存储到data对象中，以便后续使用
        data.x = inputs
        #print("Shape of data.x:", data.x.shape)


        # 生成标签矩阵
        decoder_edge_label_matrix = generate_edge_label_matrix(edges, num_nodes, 'st_st', 5)

        st_indices = torch.arange(len(st_node_features))
        return data, edge_index_list, decoder_edge_label_matrix, st_indices

    def len(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.edge_index_list, self.decoder_edge_label_matrix, self.st_indices






