import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class Attn_head(nn.Module):
    def __init__(self, in_channel, out_sz, in_drop, coef_drop, activation, residual):
        super(Attn_head, self).__init__()
        self.in_drop = nn.Dropout(in_drop)
        self.coef_drop = nn.Dropout(coef_drop)
        self.activation = activation
        self.residual = residual

        self.conv_seq = nn.Conv1d(in_channel, out_sz, 1, bias=False)
        self.conv_f1 = nn.Conv1d(out_sz, 1, 1)
        self.conv_f2 = nn.Conv1d(out_sz, 1, 1)
        self.layer_norm = nn.LayerNorm(out_sz)
        self.bias = nn.Parameter(torch.zeros(out_sz))
        #print("in_channel in Attn_head:", in_channel)
        #print("out_sz in Attn_head:", out_sz)

        # 如果有残差连接，确保输入和输出尺寸相同，或者有一个1x1卷积来更改输入尺寸
        if self.residual:
            if in_channel != out_sz:
                self.residual_conv = nn.Conv1d(in_channel, out_sz, 1)
            else:
                self.residual_conv = None


    def forward(self, seq, edge_index):

        seq = seq.transpose(0, 1)
        seq = self.in_drop(seq)
        seq_fts = self.conv_seq(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)
        #分块计算单个图
        num_splits = 100
        total_edges = edge_index.shape[1]
        split_size = edge_index.shape[1] // num_splits

        results = []
        #计算注意力系数
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i != num_splits - 1 else total_edges  # 对于最后一个批次，结束索引为total_edges
            sub_edge_index = edge_index[:, start_idx:end_idx]
            start_features = f_1[0,sub_edge_index[0]]
            end_features = f_2[0,sub_edge_index[1]]
            sum_features = start_features + end_features
            e_sub = F.leaky_relu(sum_features)
            results.append(e_sub)
            #print("Length of results:", len(results))
            #print("Shape of first tensor in results:", results[0].shape)
        e = torch.cat(results, dim=0)
        #print("Shape of e:", e.shape)
        coefs = F.softmax(e, dim=0)

        # 初始化一个空的列表来保存每个批次的weighted_feats
        weighted_feats_list = []
        total_edges = edge_index.shape[1]
        split_size = total_edges // num_splits
        # 计算注意力权重
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i != num_splits - 1 else total_edges  # 对于最后一个批次，结束索引为total_edges
            # 获取当前批次的coefs和seq_fts的部分
            sub_coefs = coefs[start_idx:end_idx]
            sub_seq_fts = seq_fts[:, edge_index[1][start_idx:end_idx]].transpose(0, 1)
            # 计算当前批次的weighted_feats
            sub_weighted_feats = sub_coefs.unsqueeze(-1) * sub_seq_fts
            weighted_feats_list.append(sub_weighted_feats)
        # 合并所有批次的weighted_feats
        weighted_feats = torch.cat(weighted_feats_list, dim=0)
        weighted_feats = self.coef_drop(weighted_feats)
        vals = torch_scatter.scatter_add(weighted_feats, edge_index[0], dim=0, dim_size=seq.size(1))
        #print("Shape of vals.shape:", vals.shape)
        ret = vals + self.bias
        #print("Shape of ret1.shape:", ret.shape)

        # residual connection
        if self.residual:
            if self.residual_conv is not None:
                #print("do residual_conv")
                seq_transposed = seq.transpose(0, 1)
                seq_with_extra_dim = seq_transposed.unsqueeze(2)
                convoluted_seq = self.residual_conv(seq_with_extra_dim)
                convoluted_seq_squeezed = convoluted_seq.squeeze(2)
                ret = ret + convoluted_seq_squeezed  # 注意这里的转置操作
            else:
                #print("no residual_conv")
                if ret.shape == seq.shape:
                    ret = ret + seq
                else:
                    ret = ret + seq.transpose(0, 1)  # 注意这里的转置操作

        return self.activation(ret)


class SimpleAttLayer(nn.Module):
    def __init__(self, attention_size, hidden_size=256, return_alphas=False):
        super(SimpleAttLayer, self).__init__()
        self.return_alphas = return_alphas
        self.attention_size = attention_size
        self.hidden_size = hidden_size

        # 创建全连接层以调整输入尺寸
        self.fc = nn.Linear(self.hidden_size, hidden_size)

        # 创建注意力层的参数
        self.w_omega = nn.Parameter(torch.randn(hidden_size, self.attention_size) * 0.1)
        self.b_omega = nn.Parameter(torch.randn(self.attention_size) * 0.1)
        self.u_omega = nn.Parameter(torch.randn(self.attention_size) * 0.1)

    def forward(self, inputs):
        # 调整输入尺寸
        inputs = self.fc(inputs)

        # Fully connected layer
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        alphas = F.softmax(vu, dim=1)
        output = (inputs * alphas.unsqueeze(-1)).sum(dim=1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas

