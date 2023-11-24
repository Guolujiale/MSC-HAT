import pandas as pd
import numpy as np
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt

def elbow_method_gpu(data, k_range):
    """ 使用手肘法确定最佳簇心数量，运行在 GPU 上 """
    sse = []
    for k in k_range:
        cluster_ids_x, cluster_centers = kmeans(
            X=data, num_clusters=k, distance='euclidean', device=torch.device('cuda:0')
        )
        # 计算每个点到其簇心的距离的平方和
        distances = torch.norm(data - cluster_centers[cluster_ids_x], dim=1, p=2)
        sse.append(torch.sum(distances ** 2).item())

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, '-o')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Sum of squared distance')
    plt.title('Elbow Method For Optimal K (GPU)')
    plt.show()

def cluster_data_gpu(data, n_clusters):
    """ 在 GPU 上对数据进行聚类 """
    cluster_ids_x, cluster_centers = kmeans(
        X=data, num_clusters=n_clusters, distance='euclidean', device=torch.device('cuda:0')
    )
    return cluster_ids_x, cluster_centers

def main():
    # 加载CSV文件
    embeddings_csv = 'embedding.csv'  # 更改为你的文件名
    df = pd.read_csv(embeddings_csv)

    # 从CSV文件中提取特征，并转换为 PyTorch 张量
    features = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float).cuda()  # 假设第一列是节点索引
    print(features.shape)  # 应该是 [n_samples, n_features]
    print(features.dtype)  # 应该是 torch.float32 或 torch.float64

    # 使用手肘法确定最佳簇心数量
    elbow_method_gpu(features, range(1, 10))  # 检查1到10个簇心

    # 确定最佳簇心数量后进行聚类
    n_clusters = 8  # 更改为从手肘法中确定的最佳簇心数量
    labels, _ = cluster_data_gpu(features, n_clusters)

    # 将聚类标签附加到原始数据
    df['cluster'] = labels.cpu().numpy()
    df.to_csv('clustered_embeddings.csv', index=False)



if __name__ == '__main__':
    main()
