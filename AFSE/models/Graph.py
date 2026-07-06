from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph



def knn_graph(x, k):
    # x: [N, feature_dim]
    N = x.size(0)
    dist = torch.cdist(x, x)  # 计算所有点之间的欧氏距离
    knn_idx = dist.argsort(dim=1)[:, 1:k+1]  # 获取每个点的 k 个最近邻点索引

    adj = torch.zeros((N, N), device=x.device)
    for i in range(N):
        adj[i, knn_idx[i]] = 1
    adj = adj + adj.T  # 对称化
    adj[adj > 1] = 1  # 去除重复的边
    return adj
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, attn_weights = self.attn(x, x, x)
        return attn_weights

class GraphBlock(nn.Module):
    def __init__(self, c_out , d_model , conv_channel, skip_channel,
                        gcn_depth , dropout, propalpha  , node_dim):
        super(GraphBlock, self).__init__()
        self.start_conv = nn.Conv2d(64, conv_channel, (1, 1)) #506,1
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, 64, (1, 1 ))
        self.norm = nn.BatchNorm2d(d_model)


    def init_G_ED(self, batch):
        with torch.no_grad():
            x = batch.detach()
            x_in = x.reshape(x.size(0), x.size(2), -1).to(batch.device)
            adjs = []
            for i in range(x_in.shape[0]):
                cdist = torch.cdist(x_in[i, :, :], x_in[i, :, :])
                # 对 cdist 进行归一化
                cdist_max = cdist.max()
                cdist = cdist / cdist_max
                # 计算相关性系数
                cdist = 1 - cdist
                adjs.append(cdist)
            adj = torch.stack(adjs)
        return adj


    def forward(self, x):
        adp1 = self.init_G_ED(x)
        adp = torch.stack([F.softmax(F.relu(adp1[i]), dim=1) for i in range(adp1.size(0))])
        out = self.start_conv(x)
        batch_size = x.size(0)
        out_list = []
        for i in range(batch_size):
            out_i = self.gelu(self.gconv1(out[i].unsqueeze(0), adp[i]))
            out_list.append(out_i)
        out = torch.cat(out_list, dim=0)
        out = self.end_conv(out)
        out = self.norm(x + out)
        return out


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

#[32,32,2]
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.conv1 = nn.Linear(in_features=161, out_features=320)
        self.conv2 = nn.Linear(in_features=320, out_features=161)
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.dropout_layer = nn.Dropout(p=dropout)
    # [64,32,7,96] [7,7]

    def forward(self, x, adj):
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h=x
        ho=x
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            # h = self.conv1(h)
            h= self.alpha * h + (1 - self.alpha) * self.nconv(h, a) #[64,32,7,96]
            # h = self.alpha * x1 + (1 - self.alpha) * h
            ho = torch.cat([h,ho],dim=1) #[64,96,7,96]
        ho = self.mlp(ho)
        return ho  #[64,32,7,96]

if __name__ == '__main__':
    model=GraphBlock(101,4,32,32,2,0.1,0.1,101)
    x=torch.randn((8,4,101,161))
    y=model(x)
    print(y.shape)