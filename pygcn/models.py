import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.lambda_param=nn.Parameter(torch.FloatTensor([1]))

    def forward(self, x, adj,centrality):

        # #第一种算法
        # row_sum = adj.sum(dim=1, keepdim=True)
        # adj = adj / row_sum
        # adj = adj + centrality*self.lambda_param

        # 第二种算法
        adj=adj+centrality*self.lambda_param
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sum

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)