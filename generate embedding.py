import torch
from dataset import HeteroGraphDataset
from module import Encoder
import pandas as pd

def load_model(model_path, device, in_channel, bottle_size, attn_drop, ffd_drop, hid_units, n_heads):
    """加载训练好的编码器模型"""
    model = Encoder(in_channel=in_channel, bottle_size=bottle_size, attn_drop=attn_drop, ffd_drop=ffd_drop,
                    hid_units=hid_units, n_heads=n_heads, residual=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['encoder_state_dict'])
    return model

def remap_indices(edge_index, mapping):
    """使用映射来转换edge_index中的节点索引"""
    return torch.stack([mapping[edge_index[0]], mapping[edge_index[1]]])

def filter_edge_index(edge_index, mapping_tensor):
    """过滤和重映射边索引"""
    mask = (mapping_tensor[edge_index[0]] >= 0) & (mapping_tensor[edge_index[1]] >= 0)
    edge_index = edge_index.to(mapping_tensor.device)
    filtered_edge_index = edge_index[:, mask]
    remapped_edge_index = remap_indices(filtered_edge_index, mapping_tensor)
    return remapped_edge_index

def save_embeddings_to_batch_csv(batch_embeddings, batch_indices, output_csv_filename):
    """将单个批次的嵌入保存到CSV文件"""
    batch_embeddings_df = pd.DataFrame(batch_embeddings.numpy())
    batch_embeddings_df.insert(0, 'node_id', batch_indices.cpu().numpy())  # 插入节点序号

    # 追加到文件，如果文件不存在则创建
    with open(output_csv_filename, 'a') as f:
        batch_embeddings_df.to_csv(f, header=f.tell() == 0, index=False, lineterminator='\n')




def extract_features_in_batches(encoder, data, edge_index_list, device, batch_size, output_csv_filename):
    """分批提取特征并保存到CSV文件"""
    encoder.eval()
    num_nodes = data.x.size(0)
    data_x = data.x.to(device)

    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)
        batch_indices = torch.arange(start_idx, end_idx, device=device)
        mapping_tensor = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        mapping_tensor[batch_indices] = torch.arange(end_idx - start_idx, device=device)

        batch_data = data_x[batch_indices].to(device)
        batch_edge_index_list = [filter_edge_index(ei, mapping_tensor) for ei in edge_index_list]
        with torch.no_grad():
            batch_bottle_size, batch_features = encoder(batch_data, [ei.to(device) for ei in batch_edge_index_list])
        save_embeddings_to_batch_csv(batch_bottle_size.cpu(), batch_indices, output_csv_filename)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_filename = 'data/final_data.csv'
    model_path = 'run/epoch_578_Loss_0.3589.pt'
    output_csv_filename = 'embedding.csv'

    # 参数设定
    in_channel = 2000
    bottle_size = 200
    attn_drop = 0.2
    ffd_drop = 0.2
    hid_units = [64, 128]
    n_heads = [4, 2]
    batch_size = 1600  # 可以根据您的硬件调整批大小

    # 加载数据集
    dataset = HeteroGraphDataset(data_filename)
    data, edge_index_list, _, _ = dataset[0]

    # 加载模型
    encoder = load_model(model_path, device, in_channel, bottle_size, attn_drop, ffd_drop, hid_units, n_heads)

    # 分批提取特征并保存到CSV文件
    extract_features_in_batches(encoder, data, edge_index_list, device, batch_size, output_csv_filename)

if __name__ == '__main__':
    main()
