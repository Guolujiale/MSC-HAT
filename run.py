import torch
import torch.optim as optim
from dataset import HeteroGraphDataset
from torch.optim.lr_scheduler import CosineAnnealingLR # 引入余弦退火调度器
from sklearn.metrics import f1_score
import os
import time  # 导入time模块

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def write_log(message):
    with open('run/unsup.txt', 'a') as f:
        f.write(message + '\n')
    print(message)

def sample_data(st_indices, num_train):
    """
    从所有节点中随机选择一个子集作为训练集。
    """
    train_indices = st_indices[torch.randperm(len(st_indices))[:num_train]]
    return train_indices


def remap_edge_label_matrix(edge_label_matrix, train_indices):
    # 创建一个映射张量，其长度为edge_label_matrix中最大的索引加1
    max_index = edge_label_matrix.max() + 1
    mapping = torch.full((max_index,), -1, dtype=torch.long, device=edge_label_matrix.device)
    # 设置train_indices在映射中的新值
    mapping[train_indices] = torch.arange(train_indices.size(0), device=edge_label_matrix.device)
    # 应用映射
    remapped_edges = mapping[edge_label_matrix[:, :2]]
    # 过滤掉不在子图中的边
    mask = (remapped_edges[:, 0] >= 0) & (remapped_edges[:, 1] >= 0)
    remapped_edges = remapped_edges[mask]
    edge_labels = edge_label_matrix[mask, 2]  # 保持标签不变
    # 合并边和标签
    train_edge_label_matrix = torch.cat([remapped_edges, edge_labels.unsqueeze(1)], dim=1)
    return train_edge_label_matrix

def adjust_decoder_edge_label_matrix(edge_label_matrix, train_indices):
    """
    调整Decoder_edge_label_matrix，只包含train_indices中的节点对。
    """
    mask = torch.isin(edge_label_matrix[:, 0], train_indices) & torch.isin(edge_label_matrix[:, 1], train_indices)
    sub_edge_label_matrix = edge_label_matrix[mask]
    return sub_edge_label_matrix


def remap_indices(edge_index, mapping):
    # 使用映射来转换edge_index中的节点索引
    return torch.stack([mapping[edge_index[0]], mapping[edge_index[1]]])


def filter_edge_index(edge_index, train_indices):
    # 创建一个映射，将全图节点索引映射到子图节点索引
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices.cpu().numpy())}
    mapping_tensor = torch.tensor([mapping.get(idx, -1) for idx in range(edge_index.max() + 1)],
                                  device=edge_index.device)

    # 过滤边，确保它们完全位于子图中
    mask = torch.isin(edge_index[0], train_indices) & torch.isin(edge_index[1], train_indices)
    filtered_edge_index = edge_index[:, mask]

    # 重映射索引
    remapped_edge_index = remap_indices(filtered_edge_index, mapping_tensor)

    return remapped_edge_index


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_filename = 'data/final_data.csv'
    dataset = HeteroGraphDataset(data_filename)
    data, edge_index_list, Decoder_edge_label_matrix, st_indices= dataset[0]
    data = data.to(device)
    st_indices = st_indices.to(device)
    Decoder_edge_label_matrix = Decoder_edge_label_matrix.to(device)
    #inputs = inputs.to(device)
    edge_index_list = [edge_index.to(device) for edge_index in edge_index_list]
    for idx, edge_index in enumerate(edge_index_list):
        print(f"Shape of edge_index {idx}: {edge_index.shape}")

    # 参数设定
    bottle_size = 200
    nb_nodes_list = [data.num_nodes]

    attn_drop = 0.2
    ffd_drop = 0.2
    en_hid_units = [64,128]
    en_n_heads = [4,2]

    in_channel = 2000
    # 添加一个变量来保存最低的损失
    best_loss = float('inf')
    model_save_path = ""
    if not os.path.exists('run'):
        os.makedirs('run')

    hyperparameters = f'''
            in_channel: {in_channel}
            nb_classes: {bottle_size}
            nb_nodes_list: {nb_nodes_list}
            attn_drop: {attn_drop}
            ffd_drop: {ffd_drop}
            '''

    from module import Encoder, Decoder
    # 定义模型、损失和优化器
    Encoder = Encoder(in_channel=in_channel, bottle_size=bottle_size, attn_drop=attn_drop, ffd_drop=ffd_drop,
                         hid_units=en_hid_units, n_heads=en_n_heads, residual=True).to(device)
    Decoder = Decoder(bottle_size = bottle_size , out_channel=1).to(device)

    optimizer = optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.004, weight_decay=0.0001)
    criterion = torch.nn.BCELoss()
    # 定义余弦退火调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

    def train(encoder, decoder, data, edge_index_list, train_indices, sub_decoder_edge_label_matrix, criterion,
              optimizer):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        # 获取子集的节点特征和边索引列表
        sub_data = data.x[train_indices]
        sub_edge_index_list = [filter_edge_index(edge_index, train_indices) for edge_index in edge_index_list]
        # 使用Encoder对子集的节点进行特征提取
        bottle_vec, node_embeddings = encoder(sub_data, sub_edge_index_list)
        # 准备Decoder的输入：提取sub_decoder_edge_label_matrix中的节点对的特征
        edge_features = torch.cat((bottle_vec[sub_decoder_edge_label_matrix[:, 0]],
                                   bottle_vec[sub_decoder_edge_label_matrix[:, 1]]), dim=1)
        # 使用Decoder预测节点对之间是否存在边
        # 使用Decoder预测节点对之间是否存在边
        edge_predictions = decoder(edge_features)  # Decoder只接收边特征作为输入
        actual_labels = sub_decoder_edge_label_matrix[:, 2]  # 最后一列是边的实际标签
        actual_labels = actual_labels.float().unsqueeze(1)

        loss = criterion(edge_predictions, actual_labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 计算准确率
        predicted_labels = (edge_predictions > 0.5).float().squeeze()  # 将概率转换为二进制预测
        accuracy = (predicted_labels == actual_labels).float().mean().item()  # 计算准确率

        return loss.item(), accuracy

    write_log(hyperparameters)

    # 训练和测试的主循环
    num_epochs = 10000
    for epoch in range(num_epochs):
        #epoch_start_time = time.time()  # 获取epoch开始的时间戳
        if epoch % 300 == 0:
            write_log("re-sampleing train data...")
            train_indices = sample_data(st_indices, num_train=1600)
            train_indices = train_indices.to(device)
            # 将原始的edge_label_matrix重映射到基于子图的索引
            train_edge_label_matrix = remap_edge_label_matrix(Decoder_edge_label_matrix.clone(), train_indices)
            # 然后应用adjust_decoder_edge_label_matrix
            Sub_Decoder_edge_label_matrix = adjust_decoder_edge_label_matrix(train_edge_label_matrix, train_indices)
            sub_edge_index_list = [filter_edge_index(edge_index, train_indices) for edge_index in edge_index_list]

            print(f"Shape of Sub_Decoder_edge_label_matrix: {Sub_Decoder_edge_label_matrix.shape}")
            # 打印每个 sub_edge_index 的大小
            for idx, sub_edge_index in enumerate(sub_edge_index_list):
                print(f"Shape of sub_edge_index {idx}: {sub_edge_index.shape}")

        loss, accuracy = train(Encoder, Decoder, data, edge_index_list, train_indices, Sub_Decoder_edge_label_matrix, criterion, optimizer)
        scheduler.step()
        write_log(f"Epoch: {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        if loss < best_loss:
            best_loss = loss
            model_save_path = f"run/epoch_{epoch + 1}_Loss_{best_loss:.4f}.pt"
            torch.save({'encoder_state_dict': Encoder.state_dict(), 'decoder_state_dict': Decoder.state_dict()},
                       model_save_path)
            write_log(f"New best training loss: {loss:.4f}, Model saved: {model_save_path}")

if __name__ == '__main__':
    main()