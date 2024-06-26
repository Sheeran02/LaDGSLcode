import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator
class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, params, concat=True):

        super(GAT, self).__init__()
        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.out_features = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn
        self.num_nodes = params.num_nodes
        self.device = params.device
        self.add_transe_emb = params.add_transe_emb
        self.concat = concat

        self.alpha = 0.01#瞎设置的

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        self.W = nn.Parameter(torch.zeros(size=(params.inp_dim, params.emb_dim)))#权重矩阵 大小好像有点问题，原代码给的是in_feature和out_feature，不确定我这个对不对
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*params.emb_dim, 1)))#问题同上
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, input, adj):
        # input: 输入特征矩阵，大小为[N, in_features]
        # adj: 邻接矩阵，大小为[N, N]
        #print('input=', input.shape)
        # 第一步：矩阵乘法
        h = torch.mm(input, self.W)  # h的大小为[N, out_features]

        # 第二步：构造注意力机制的输入
        N = h.size()[0]  # 图中节点数目
        # 将h分别复制N次和1次，然后拼接成一个(N*N, 2*out_features)的矩阵，最后reshape成[N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # 对注意力机制的输入做线性变换和激活函数操作
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # e的大小为[N, N]

        # 第三步：利用邻接矩阵构造注意力矩阵
        # 首先构造一个全零的矩阵，大小为[N, N]
        zero_vec = -9e15 * torch.ones_like(e)
        zero_vec = zero_vec.cuda()
        # print(adj.shape)
        # print(e.shape)
        # print(zero_vec.shape)
        # 利用邻接矩阵，将需要注意的位置的值替换成e中的值
        attention = torch.where(adj > 0, e, zero_vec)
        # 对每个节点的注意力值做softmax归一化
        attention = F.softmax(attention, dim=1)
        # 对注意力矩阵做dropout操作02
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 第四步：利用注意力矩阵和节点特征矩阵，计算节点更新后的特征矩阵
        h_prime = torch.matmul(attention, h)  # h_prime的大小为[N, out_features]

        # 第五步：根据是否使用拼接操作，返回更新后的节点特征矩阵
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

