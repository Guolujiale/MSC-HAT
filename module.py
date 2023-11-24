import torch.nn as nn
from layer import Attn_head, SimpleAttLayer
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channel, bottle_size, attn_drop, ffd_drop,
                 hid_units, n_heads, activation=nn.ELU(), residual=False, mp_att_size=64,nn_liner_batch_size=1600):
        super(Encoder, self).__init__()

        if len(hid_units) != len(n_heads):
            raise ValueError("The lengths of hid_units and n_heads must be the same.")

        self.bottle_vec = bottle_size
        #self.nb_nodes_list = nb_nodes_list
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.mp_att_size = mp_att_size
        self.in_channel = in_channel
        self.nn_liner_batch_size= nn_liner_batch_size
        print("in_channel provided to Encoder:", in_channel)

        self.attn_layers = nn.ModuleList()
        in_sz = self.in_channel

        for i in range(len(hid_units)):
            self.attn_layers.append(
                nn.ModuleList([Attn_head(in_channel=in_sz, out_sz=hid_units[i], activation=self.activation,
                                         in_drop=self.ffd_drop,
                                         coef_drop=self.attn_drop, residual=self.residual)
                               for _ in range(n_heads[i])]
                              ))
            in_sz = n_heads[i] * hid_units[i]

        self.simpleAttLayer = SimpleAttLayer(self.mp_att_size)
        self.final_layers = nn.ModuleList([nn.Linear(self.hid_units[-1] * self.n_heads[-1], bottle_size) for _ in range(n_heads[-1])])
#处理分批全连接层
    def forward_with_minibatching(self, final_embed, batch_size):
        num_nodes = final_embed.shape[0]
        outputs = []

        # 遍历每个小批次
        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)
            mini_batch = final_embed[start_idx:end_idx]
            #print("end_idx:", end_idx)
            # 对这个小批次应用全连接层
            mini_out = [layer(mini_batch) for layer in self.final_layers]
            outputs.append(sum(mini_out) / len(mini_out))

            # 释放mini_out的内存
            del mini_out
            torch.cuda.empty_cache()

        outputs_gpu = [output.cuda() for output in outputs]
        # 将所有的小批次结果拼接起来
        logits = torch.cat(outputs_gpu, dim=0)
        # 释放outputs的内存
        del outputs
        torch.cuda.empty_cache()
        return logits

    def forward(self, data, edge_index_list):
        inputs = data
        #print("Checking inputs in module4.py")
        #print(inputs)
        #print(type(inputs))

        embed_list = []

        for edge_index in edge_index_list:
            #第一个隐藏层的n_heads[0]个注意力头的计算
            attns = [self.attn_layers[0][j](inputs, edge_index) for j in range(self.n_heads[0])]
            h_1_new = torch.cat(attns, dim=-1)
            #其他隐藏层的注意力头的计算
            for i in range(1, len(self.hid_units)):
                if i != 1:  # 释放前一个h_1的内存
                    del h_1
                    torch.cuda.empty_cache()

                h_1 = h_1_new
                attns = [self.attn_layers[i][j](h_1, edge_index) for j in range(self.n_heads[i])]
                h_1_new = torch.cat(attns, dim=-1)
                #print("Shape of hide_h_1.shape:", h_1.shape)
            # h_1.unsqueeze(1):[num_nodes,1,hid_units[-1]×n_heads[-1]]
            embed_list.append(h_1_new.unsqueeze(1))
            #[6,num_nodes,1,hid_units[-1]×n_heads[-1]]

        multi_embed = torch.cat(embed_list, dim=1)#[num_nodes,6,hid_units[-1]×n_heads[-1]]
        #print("Shape of multi_embed.shape:", multi_embed.shape)
        final_embed = self.simpleAttLayer(multi_embed)
        #final_embed对multi_embed的第二个维度（即6个异构子图的输出）进行加权平均后得到的: [num_nodes,hid_units[-1]×n_heads[-1]]
        #att_val是注意力权重，它的形状应该是[num_nodes, 6]
        #print("Shape of final_embed.shape:", final_embed.shape)
        #print("Shape of att_val.shape:", att_val.shape)

        # 使用分批处理计算 logits
        bottle_vec = self.forward_with_minibatching(final_embed, self.nn_liner_batch_size)

        return bottle_vec, final_embed

class Decoder(nn.Module):
    def __init__(self, bottle_size, out_channel):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2 * bottle_size, out_channel)

    def forward(self, edge_features):
        # edge_features: 从Encoder获得的拼接的边特征
        # 应用全连接层和sigmoid激活
        edge_pred = torch.sigmoid(self.fc(edge_features))
        return edge_pred
