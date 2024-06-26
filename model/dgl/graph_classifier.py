from .rgcn_model import RGCN
from .gat_model import GAT
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):params：命名元组，包括输入维度、隐藏层维度、关系嵌入维度、输出维度、关系数量和基的数量
        super().__init__()

        self.params = params
        self.dropout = nn.Dropout(p = params.dropout)
        self.relu = nn.ReLU()
        self.train_rels = params.train_rels
        self.relations = params.num_rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.gat = GAT(params)

        # MLP
        self.mp_layer1 = nn.Linear(self.params.feat_dim, 256)  # 多层感知机模型的第一层线性层
        self.mp_layer2 = nn.Linear(256, self.params.emb_dim)  # 多层感知机模型的第二层线性层
        self.bn1 = nn.BatchNorm1d(256)  # 批标准化层

        # Decoder
        #全连接层用于特征融合
        # 判断是否需要添加头实体嵌入（head entity embedding）和尾实体嵌入（tail entity embedding）
        if self.params.add_ht_emb and self.params.add_sb_emb:
            # 判断是否需要添加特征嵌入（feature embedding）和转换嵌入（transE embedding）
            # params.emb_dim=params.emb_dim*2
            if self.params.add_feat_emb and self.params.add_transe_emb:
                # self.fc_layer = nn.Linear(3 * (1 + self.params.num_gcn_layers) * (
                #             self.params.emb_dim + self.params.inp_dim) + 2 * self.params.emb_dim, 512)
                self.fc_layer = nn.Linear(1808, 512)
            elif self.params.add_feat_emb:
                self.fc_layer = nn.Linear(
                    3 * (self.params.num_gcn_layers) * self.params.emb_dim + 2 * self.params.emb_dim, 512)
            else:
                self.fc_layer = nn.Linear(
                    3 * (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim), 512)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1 + self.params.num_gcn_layers) * self.params.emb_dim, 512)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, 512)
        # 设置两个额外的全连接层，分别将输入维度512降为128，和128降为1
        self.fc_layer_1 = nn.Linear(512, 128)
        self.fc_layer_2 = nn.Linear(128, 1)
        print('fc',self.fc_layer)
    def omics_feat(self, emb):
        self.genefeat = emb

    def get_omics_features(self, ids):
        a = []
        for i in ids:
            a.append(self.genefeat[i.cpu().numpy().item()])
        return np.array(a)


    def forward(self, data):
        #该模型以图g为输入，并对其进行多次操作以获得预测。
        g = data
        g.ndata['h'] = self.gnn(g)
        # print(g.ndata['h'])
        g1_out = mean_nodes(g, 'repr')
        #print('g_out.shape=',g1_out.shape)
        g2 = data
        nx_g2 = g2.to_networkx()
        adj_matrix = nx.adjacency_matrix(nx_g2)
        adj_tensor = torch.tensor(adj_matrix.todense(), dtype=torch.float32).cuda()
        g2.ndata['h'] = self.gat(g2.ndata['feat'], adj_tensor)
        g2_out = mean_nodes(g2, 'repr')
        #print('g2_out.shape=', g2_out.shape)
        g_out = torch.cat([g1_out, g2_out], dim=2)
        #从节点数据字典中获取头结点的索引并保存在变量 head_ids 中
        # 获取头结点的表示，并将结果保存在变量 head_embs 中

        #这一段不知道应不应该聚合
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs1 = g.ndata['repr'][head_ids]
        head_embs2 = g2.ndata['repr'][head_ids]
        head_embs=torch.cat([head_embs1,head_embs2], dim=2)
        #同上操作tail节点
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs1 = g.ndata['repr'][tail_ids]
        tail_embs2 = g2.ndata['repr'][tail_ids]
        tail_embs = torch.cat([tail_embs1, tail_embs2], dim=2)
        #调用 get_omics_features 方法获取头结点和尾节点的基因表达特征向量，将结果保存在 head_feat 和 tail_feat 中，并转换为 pytorch 的 FloatTensor 格式，并将其移到指定的计算设备上（如 GPU）
        head_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][head_ids])).to(self.params.device)
        tail_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][tail_ids])).to(self.params.device)

        if self.params.add_feat_emb:
            fuse_feat1 = self.mp_layer2(self.bn1(self.relu(self.mp_layer1(head_feat))))
            fuse_feat2 = self.mp_layer2(self.bn1(self.relu(self.mp_layer1(tail_feat))))
            fuse_feat = torch.cat([fuse_feat1, fuse_feat2], dim=1)

        # a1=g_out.view(-1, (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
        # a2=head_embs.view(-1, (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
        # a3=tail_embs.view(-1, (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
        # a4=fuse_feat.view(-1, 2 * self.params.emb_dim)
        # print(g_out.shape)
        # print(head_embs.shape)
        # print(tail_embs.shape)
        # print(fuse_feat.shape)
        # print("g_out=",a1)
        # print("head=", a2)
        # print("tail=", a3)
        # print("fuse=", a4)
        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                g_rep = torch.cat(
                    [g_out.view(-1, 560),
                     head_embs.view(-1,560),
                     tail_embs.view(-1, 560),
                     fuse_feat.view(-1, 2 * self.params.emb_dim)
                     ], dim=1)
            elif self.params.add_feat_emb:
                # g_rep = torch.cat([g_out.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                #                    head_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                #                    tail_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                #                    fuse_feat.view(-1, 2 * self.params.emb_dim)
                #                    ], dim=1)
                g_rep = torch.cat([g_out.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim*2),
                                   head_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   fuse_feat.view(-1, 2 * self.params.emb_dim)
                                   ], dim=1)
            else:
                g_rep = torch.cat(
                    [g_out.view(-1, (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                     head_embs.view(-1, (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                     tail_embs.view(-1, (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                     # fuse_feat.view(-1, 2*self.params.emb_dim)
                     ], dim=1)

        elif self.params.add_ht_emb:
            g_rep = torch.cat([
                head_embs.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim),
                tail_embs.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim)
            ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim*2)
       # print('g_rep.shape=', g_rep.shape)

        output = self.fc_layer_2(self.relu(self.fc_layer_1(self.relu(self.fc_layer(self.dropout(g_rep))))))
        output = output.squeeze(-1)#加了squeeze之后迭代速度增快了
        return (output, g_rep)
