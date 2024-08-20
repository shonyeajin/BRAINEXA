import torch, gc
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import os
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool
from torch_geometric.utils import softmax
import argparse
import os
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
import torch_geometric.transforms as T
import random
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pad_sequence
import subprocess
import json
import pprint
import time
from torch_geometric.utils import to_networkx
from datetime import datetime
import scipy.stats as stats
from torch import Tensor
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import utils

def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = torch.sigmoid(gumbels)

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1], indices[2]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, y_soft



# model
scalar = 20
eps = 1e-10


class Explainer_MLP(torch.nn.Module):
    def __init__(self, num_features, dim, n_layers):
        super(Explainer_MLP, self).__init__()

        self.n_layers = n_layers
        self.mlps = torch.nn.ModuleList()

        for i in range(n_layers):
            if i:
                nn = Sequential(Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim))
            self.mlps.append(nn)

        self.final_mlp = Linear(dim, 1)


    def forward(self, x, edge_index, batch):

        for i in range(self.n_layers):
            x = self.mlps[i](x)
            x = F.relu(x)

        node_prob = self.final_mlp(x)
        node_prob = softmax(node_prob, batch)
        return node_prob


class Explainer_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, readout):
        super(Explainer_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.readout = readout

        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)

        if self.readout == 'concat':
            self.mlp = Linear(dim * num_gc_layers, 1)
        else:
            self.mlp = Linear(dim, 1)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            if i != self.num_gc_layers - 1:
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
            else:
                x = self.convs[i](x, edge_index)
            xs.append(x)

        if self.readout == 'last':
            node_prob = xs[-1]
        elif self.readout == 'concat':
            node_prob = torch.cat([x for x in xs], 1)
        elif self.readout == 'add':
            node_prob = 0
            for x in xs:
                node_prob += x

        node_prob = self.mlp(node_prob)
        node_prob = softmax(node_prob, batch)
        return node_prob


class Explainer_HGNN(torch.nn.Module):
    def __init__(self, input_dim, input_dim_edge, hidden_dim, num_gc_layers):
        super(Explainer_HGNN, self).__init__()

        self.num_node_features = input_dim
        if input_dim_edge:
            self.num_edge_features = input_dim_edge
            self.use_edge_attr = True
        else:
            self.num_edge_features = input_dim
            self.use_edge_attr = False
        self.nhid = hidden_dim
        self.num_convs = num_gc_layers
        self.convs = self.get_convs()

        self.mlp = Linear(hidden_dim*num_gc_layers, 1)

    def get_convs(self):

        convs = torch.nn.ModuleList()

        for i in range(self.num_convs):

            if i == 0:
                conv = HypergraphConv(self.num_edge_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)

            convs.append(conv)

        return convs


    def forward(self, x, edge_index, edge_attr, batch):

        if not self.use_edge_attr:
            a_, b_ = x[edge_index[0]], x[edge_index[1]]
            edge_attr = (a_ + b_) / 2
        hyperedge_index, edge_batch = DHT(edge_index, batch)

        xs = []
        # Message Passing
        for _ in range(self.num_convs):
            edge_attr = F.relu( self.convs[_](edge_attr, hyperedge_index))
            xs.append(edge_attr)

        edge_prob = self.mlp(torch.cat(xs, 1))
        edge_prob = softmax(edge_prob, edge_batch)

        return edge_prob



class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling, readout):
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout

        self.convs = torch.nn.ModuleList()
        self.dim = dim
        self.pool = self.get_pool()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)

            self.convs.append(conv)

    def forward(self, x, edge_index, batch, node_imp):

        if node_imp is not None:
            out, _ = torch_scatter.scatter_max(torch.reshape(node_imp.detach(), (1, -1)), batch)
            out = out.reshape(-1, 1)
            out = out[batch]
            node_imp /= out + eps 
            node_imp = (2 * node_imp - 1)/(2 * scalar) + 1 
            x = x * node_imp

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)


        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)

        return graph_emb, torch.cat(xs, 1), xs[-1]

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool


class GIN_DECODER(torch.nn.Module):
    def __init__(self, dim, num_features, num_gc_layers):
        super(GIN_DECODER, self).__init__()

        self.num_gc_layers = num_gc_layers


        self.convs = torch.nn.ModuleList()
        self.dim = dim

        for i in range(num_gc_layers):
            if i < num_gc_layers - 1:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, num_features))
            conv = GINConv(nn)

            self.convs.append(conv)

    def forward(self, x, edge_index):
        xs = []
        for i in range(self.num_gc_layers):
            if i < self.num_gc_layers - 1:
                x = F.relu(self.convs[i](x, edge_index))
            else:
                x = self.convs[i](x, edge_index)
            xs.append(x)

        return xs[-1]




class HyperGNN(torch.nn.Module):

    def __init__(self, input_dim, input_dim_edge, hidden_dim, num_gc_layers, pooling, readout, n_time, dropout=0.5):

        super(HyperGNN, self).__init__()

        self.num_node_features = input_dim
        if input_dim_edge:
            self.num_edge_features = input_dim_edge
            self.use_edge_attr = True
        else:
            self.num_edge_features = input_dim
            self.use_edge_attr = False
        self.nhid = hidden_dim
        self.enhid = hidden_dim
        self.num_convs = num_gc_layers
        self.pooling = pooling
        self.readout = readout
        self.n_time = n_time
        self.convs = self.get_convs()
        self.pool = self.get_pool()

        padding = ((args.temporal_kernel_size-1)//2, 0)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(self.nhid),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.nhid,
                self.nhid,
                (args.temporal_kernel_size, 1),
                (1,1),
                padding,
            ),
            nn.BatchNorm2d(self.nhid),
            nn.Dropout(dropout, inplace=True)
        )

    def forward(self, x, edge_index, edge_attr, batch, edge_imp, batch_size):
        hyperedge_indexs = []
        edge_batches = []
        latent_edge_attr = []
        for t_idx in range(self.n_time):
            hyperedge_index, edge_batch = DHT(edge_index[t_idx], batch)
            hyperedge_indexs.append(hyperedge_index)
            edge_batches.append(edge_batch)
            latent_edge_attr.append(edge_attr[t_idx])
        xs = []
        for _ in range(self.num_convs):
            spatial_embs = []
            for t_idx in range(self.n_time):
                spatial_embs.append(self.convs[_](latent_edge_attr[t_idx], hyperedge_indexs[t_idx]))
            spatial_embs = torch.stack(spatial_embs, dim=0)
            spatial_embs = spatial_embs.view(spatial_embs.size(0), batch_size, -1,spatial_embs.size(2)) # (time, batch, 160*160/100, feature)
            spatial_embs = spatial_embs.permute(1, 3, 0, 2).contiguous()
            spatial_embs = F.relu(self.tcn(spatial_embs))
            spatial_embs = spatial_embs.permute(2,0,3,1).contiguous()
            spatial_embs = spatial_embs.view(spatial_embs.size(0), -1, spatial_embs.size(3))
            for t_idx in range(self.n_time):
                latent_edge_attr[t_idx]=spatial_embs[t_idx]
            xs.append(torch.stack(latent_edge_attr, dim=0))
        xs_temp =torch.stack(xs, dim=0)
        xs_temp = xs_temp.permute(1,0,2,3).contiguous()
        
        graph_embs=[]
        for t_idx in range(self.n_time):
            if self.readout == 'last':
                graph_embs.append(self.pool(xs_temp[t_idx][-1], edge_batches[t_idx]))
            elif self.readout == 'concat':
                graph_embs.append(torch.cat([self.pool(x, edge_batches[t_idx]) for x in xs_temp[t_idx]],1))
            elif self.readout == 'add':
                graph_emb = 0
                for x in xs_temp[t_idx]:
                    graph_emb += self.pool(x, edge_batches[t_idx])
                graph_embs.append(graph_emb)
        graph_embs = torch.stack(graph_embs, dim=0)

        return graph_embs, None, torch.stack(latent_edge_attr, dim=0)



    def get_convs(self):
        convs = torch.nn.ModuleList()
        for i in range(self.num_convs):
            if i == 0:
                conv = HypergraphConv(self.num_edge_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)
            convs.append(conv)

        return convs

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))

        return pool



class HyperGNN_DECODER(torch.nn.Module):

    def __init__(self, hidden_dim, input_dim, input_dim_edge, num_gc_layers,):

        super(HyperGNN_DECODER, self).__init__()

        self.num_node_features = input_dim
        if input_dim_edge:
            self.num_edge_features = input_dim_edge
            self.use_edge_attr = True
        else:
            self.num_edge_features = input_dim
            self.use_edge_attr = False
        self.nhid = hidden_dim
        self.enhid = hidden_dim
        self.num_convs = num_gc_layers
        self.convs = self.get_convs()


    def forward(self, x, edge_index, edge_attr, batch):
        
        edge_attr = edge_attr.unsqueeze(1)
        edge_attr = edge_attr.to(dtype=torch.float32)

        hyperedge_index, edge_batch = DHT(edge_index, batch)

        xs = []
        
        for _ in range(self.num_convs):
            if _ < self.num_convs - 1:
                edge_attr = F.relu( self.convs[_](edge_attr, hyperedge_index))
            else:
                edge_attr = self.convs[_](edge_attr, hyperedge_index)
            xs.append(edge_attr)

        return xs[-1]

    def get_convs(self):
        convs = torch.nn.ModuleList()
        for i in range(self.num_convs):
            if i == self.num_convs -1:
                conv = HypergraphConv(self.nhid, self.num_node_features)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)
            convs.append(conv)

        return convs


class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # kernel_size = 1
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):  # x.shape: torch.Size([64, 64, 128, 22])
        assert A.size(0) == self.kernel_size
        x = self.conv(x)  # torch.Size([64, 64, 128, 22])  see as: N*C*H*W
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)  # torch.Size([64, 1, 64, 128, 22])
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # torch.Size([64, 64, 128, 22])
        return x.contiguous(), A  # x.contiguous().shape: torch.Size([64, 64, 128, 22])

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # (11, 1)
                 n_time,
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  # padding = (5, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, n_time)  # kernel_size[1] = 1

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),  # kernel_size[0] = 11
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, A):
        res = self.residual(x)
        batch_size = A.size(0)
        xs = []
        for i in range(batch_size):
            ret, _ = self.gcn(x[i].unsqueeze(0), A[i])
            xs.append(ret)
        x = torch.cat(xs, dim = 0)
        x = self.tcn(x) + res
        return self.relu(x), A
    
class STGCN_encoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` wheretorch.nn
            :math:`N` is batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self, num_features, dim, num_gc_layers, temporal_kernel_size,edge_importance_weighting, n_time, device, pooling, readout, **kwargs) -> object:
        super().__init__()
        self.pooling = pooling
        self.readout = readout
        self.pool = self.get_pool()

        kernel_size = (temporal_kernel_size, 1)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(num_features, dim, kernel_size, n_time, 1, residual=False, **kwargs),
            st_gcn(dim, dim, kernel_size, n_time, 1, residual=False, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x, edge_index, adj, batch):
        xs = []
        x = x.permute(0, 3, 1, 2).contiguous() # (batch, time, node, feature) -> (batch*M, features, time, node)
        adj = torch.tensor(adj, dtype=torch.float32, requires_grad=False).to(device) # (batch, time, node, node) = (16, 9, 160, 160)
        # for gcn, importance in zip (self.st_gcn_networks, self.edge_importance):
        #     x, _ = F.relu(gcn(x, adj))
        #     xs.append(x.permute(0, 2, 3 ,1).contiguous().view(-1, x.size(1)))
        for gcn_idx, gcn in enumerate(self.st_gcn_networks):
            x, _ = gcn(x, adj)
            xs.append(x.permute(0, 2, 3 ,1).contiguous().view(-1, x.size(1)))

            

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)
        
        return graph_emb, torch.cat(xs, 1), xs[-1]
    
    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool



class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8., qkv_bias = False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape 
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_biase))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) 

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        
    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position/np.power(10000, 2 * (hid_j//2)/d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshape = x.contiguous().view(batch_size * seq_len, -1)
        y = self.module(x_reshape)
        y = y. contiguous().view(batch_size, seq_len, -1)
        return y
    
class LSTMForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMForecastingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.time_distributed = TimeDistributed(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.time_distributed(out)

        return out



class SIGNET(nn.Module):
    def __init__(self, n_node, n_time,input_dim, input_dim_edge, args, device):
        super(SIGNET, self).__init__()

        self.device = device

        self.embedding_dim = args.hidden_dim
        if args.readout == 'concat':
            self.embedding_dim *= args.encoder_layers

        if args.explainer_model == 'mlp':
            self.explainer = Explainer_MLP(input_dim, args.explainer_hidden_dim, args.explainer_layers)
        else:
            self.explainer = Explainer_GIN(input_dim, args.explainer_hidden_dim,
                                           args.explainer_layers, args.explainer_readout)
            
        self.encoder = STGCN_encoder(input_dim, args.hidden_dim, args.encoder_layers, args.temporal_kernel_size, args.edge_importance_weighting, n_time, device, args.pooling, args.readout)
        self.encoder_hyper = HyperGNN(input_dim, input_dim_edge, args.hidden_dim, args.encoder_layers, args.pooling, args.readout, n_time)

        
        self.lstm = MyLSTM(self.embedding_dim, self.embedding_dim, args.lstm_layers)
        self.lstm_hyper = MyLSTM(self.embedding_dim, self.embedding_dim, args.lstm_layers)

        self.spatial_proj_head_p0 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.spatial_proj_head_hyper_p0 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.temporal_proj_head_p0 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.temporal_proj_head_hyper_p0 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.spatial_proj_head_p1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh())
        self.spatial_proj_head_hyper_p1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh())
        self.temporal_proj_head_p1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh())
        self.temporal_proj_head_hyper_p1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh())

        self.decoder = GIN_DECODER(args.hidden_dim, input_dim, args.decoder_layers)
        self.decoder_hyper = HyperGNN_DECODER(args.hidden_dim, input_dim, input_dim_edge, args.decoder_layers)
        
        self.forecast_decoder = LSTMForecastingModel(args.hidden_dim, args.hidden_dim, input_dim, args.lstm_num_layers)
        self.forecast_decoder_hyper = LSTMForecastingModel(args.hidden_dim, args.hidden_dim, input_dim, args.lstm_num_layers)

        self.n_node = n_node
        self.n_time = n_time
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, self.n_node * n_time, input_dim))
        self.get_token_probs = nn.Sequential(
            Block(dim=input_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                  drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm, init_values=0.),
                  nn.Linear(input_dim, 1),
                  torch.nn.Flatten(start_dim=1),
        )
        self.softmax = nn.Softmax(dim=-1)
        
        self.pos_embed = get_sinusoid_encoding_table(self.n_node * n_time, input_dim)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0.0)
                m.weight.data.fill_(1.0)

    def scale_input(self, input_tensor):
        min_input = torch.min(input_tensor)
        max_input = torch.max(input_tensor)
        normalized = (input_tensor - min_input) / (max_input - min_input)
        scaled = 2 * normalized - 1
        return scaled

    def forward(self, data, bottleneck_count):
        data_batch_temp = data.batch
        batch_size = data.label.size(0)
        data.signal = data.signal.view(-1, data.signal.shape[2]) 
        data.edge_index = data.edge_index.permute(1,0,2).contiguous() 
        data.edge_index = data.edge_index.view(data.edge_index.shape[0], -1) 
        data.fc = data.fc.view(data.fc.shape[0]*data.fc.shape[1]) 
        data.batch = np.repeat(np.arange(batch_size*self.n_time), 116)
        data.batch = torch.tensor(data.batch).to(self.device)

        x_ada = data.signal.view(batch_size, -1, data.signal.shape[1]).clone() 
        x_ada = x_ada + self.pos_embed_probs.type_as(x_ada).to(x_ada.device).clone()
        logits = self.get_token_probs(x_ada)
        logits = torch.nan_to_num(logits).view(batch_size, self.n_time, self.n_node)
        logit = []
        p_x = torch.tensor(logits)
        
        save_list =[]
        vis_mask, p_x_soft = gumbel_sigmoid(p_x, hard=True, tau=0.2)
        msk_mask = vis_mask
        for aaa in range(msk_mask.size(0)):
            for bbb in range(msk_mask.size(1)):
                bottleneck_count+=msk_mask[aaa][bbb]
        vis_mask = 1 - vis_mask
        msk_mask = msk_mask.to(torch.bool)
        vis_mask = vis_mask.to(torch.bool)

        p_x_edge = self.lift_node_score_to_edge_score(torch.flatten(p_x_soft, 0).unsqueeze(-1), data.edge_index)
        p_x_edge = p_x_edge.view(batch_size, self.n_time, -1) 
        n_edge = p_x_edge.shape[2]

        vis_mask_edge, p_x_edge_soft = gumbel_sigmoid(p_x_edge, hard=True, tau=0.2)
        msk_mask_edge = vis_mask_edge
        vis_mask_edge = 1 - vis_mask_edge
        msk_mask_edge = msk_mask_edge.to(torch.bool)
        vis_mask_edge = vis_mask_edge.to(torch.bool)


        del x_ada
        x_ada = data.signal.clone().view(batch_size, self.n_time, self.n_node, -1) 

        log_probs = p_x_soft
        log_probs[msk_mask] = 0.0

        log_probs_edge = p_x_edge_soft
        log_probs_edge[msk_mask_edge] = 0.0

        msk_mask_expanded = msk_mask.unsqueeze(-1)
        x_ada_msk = torch.where(msk_mask_expanded, torch.zeros_like(x_ada), x_ada)
        vis_mask_expanded = vis_mask.unsqueeze(-1)
        x_ada_vis = torch.where(vis_mask_expanded, torch.zeros_like(x_ada), x_ada)
    
        x_ada_vis = x_ada_vis.clone().detach()
        x_ada_msk = x_ada_msk.clone().detach()
        
        # for testing
        y, _, emb_vis = self.encoder(x_ada_vis, data.edge_index, data.adj, data.batch)
        y_masked, _, emb_msk = self.encoder(x_ada_msk, data.edge_index, data.adj, data.batch)

        y_recon = self.decoder(emb_vis, data.edge_index)
        
        # for return
        y_recon = y_recon.view(batch_size, self.n_time, self.n_node, -1)
        y_recon_target = x_ada

        # forecasting
        emb_vis = emb_vis.view(batch_size, self.n_time, self.n_node, -1).permute(0,2,1,3).contiguous()
        lstm_hidden_dim = emb_vis.shape[3]
        emb_vis = emb_vis.view(-1, self.n_time,lstm_hidden_dim)
        forecasted_value = self.forecast_decoder(emb_vis)


        del data.batch

        signal_2 = data.signal.clone().view(batch_size, self.n_time, self.n_node, -1)
        signal_2 = signal_2.permute(1,0,2,3).contiguous() 
        signal_2 = signal_2.view(self.n_time, -1, signal_2.shape[3]) 
        data.edge_index = data.edge_index.view(2, self.n_time, -1) 
        data.edge_index = data.edge_index.permute(1,0,2).contiguous() 
        data.fc = data.fc.view(self.n_time, -1)

        data.batch = data_batch_temp

        x_ada_edge = []
        for i in range(self.n_time):
            a_, b_ = signal_2[i][data.edge_index[i][0]], signal_2[i][data.edge_index[i][1]]
            a_b_ = (a_ + b_) / 2
            x_ada_edge.append(a_b_)
        x_ada_edge = torch.stack(x_ada_edge, 0) 

        msk_mask_edge = msk_mask_edge.permute(1,0,2).contiguous().view(self.n_time, -1)
        vis_mask_edge = vis_mask_edge.permute(1,0,2).contiguous().view(self.n_time, -1)
        msk_mask_edge_expanded = msk_mask_edge.unsqueeze(-1)
        vis_mask_edge_expanded = vis_mask_edge.unsqueeze(-1)

        x_ada_msk_edge = torch.where(msk_mask_edge_expanded, torch.zeros_like(x_ada_edge), x_ada_edge)
        x_ada_vis_edge = torch.where(vis_mask_edge_expanded, torch.zeros_like(x_ada_edge), x_ada_edge)

        x_ada_msk_edge = x_ada_msk_edge.clone().detach()
        x_ada_vis_edge = x_ada_vis_edge.clone().detach()

        y_hyper, _, emb_hyper_vis = self.encoder_hyper(None, data.edge_index, x_ada_vis_edge, data.batch, None, batch_size)
        y_hyper_masked, _, emb_hyper_masked = self.encoder_hyper(None, data.edge_index, x_ada_msk_edge, data.batch, None, batch_size)
        y_hyper_recon = []
        for i in range(self.n_time):
            y_hyper_recon.append(self.decoder_hyper(None, data.edge_index[i], emb_hyper_vis[i], data.batch))
        y_hyper_recon = torch.stack(y_hyper_recon, dim =0)
        
        # for return
        y_hyper_recon_target = x_ada_edge

        y_hyper = y_hyper.view(-1, y_hyper.shape[2]) 
        y_hyper_masked=y_hyper_masked.view(-1, y_hyper_masked.shape[2]) 

        # for forecasting
        emb_hyper_vis = emb_hyper_vis.permute(1,0,2).contiguous()
        forecasted_value_hyper = self.forecast_decoder_hyper(emb_hyper_vis)


        y_s_p0 = self.spatial_proj_head_p0(y) # view 1 visible
        y_hyper_s_p0 = self.spatial_proj_head_hyper_p0(y_hyper) # view 2 visible
        y_masked_s_p0 = self.spatial_proj_head_p0(y_masked) # view 1 masked
        y_hyper_masked_s_p0 = self.spatial_proj_head_hyper_p0(y_hyper_masked) # view 2 masked

        y_s_p1 = self.spatial_proj_head_p1(y) # view 1 visible
        y_hyper_s_p1 = self.spatial_proj_head_hyper_p1(y_hyper) # view 2 visible

        y = y.view(self.n_time, batch_size, -1).permute(1, 0, 2) # batch, time, hidden
        y_hyper = y_hyper.view(self.n_time, batch_size, -1).permute(1, 0, 2)
        y_masked = y_masked.view(self.n_time, batch_size, -1).permute(1, 0, 2)
        y_hyper_masked = y_hyper_masked.view(self.n_time, batch_size, -1).permute(1, 0, 2)

        y_t = self.lstm(y)
        y_hyper_t = self.lstm_hyper(y_hyper)
        y_masked_t = self.lstm(y_masked)
        y_hyper_masked_t = self.lstm_hyper(y_hyper_masked)

        y_t_p0 = self.temporal_proj_head_p0(y_t)
        y_hyper_t_p0 = self.temporal_proj_head_hyper_p0(y_hyper_t)
        y_masked_t_p0 = self.temporal_proj_head_p0(y_masked_t)
        y_hyper_masked_t_p0 = self.temporal_proj_head_hyper_p0(y_hyper_masked_t)

        y_t_p1 = self.temporal_proj_head_p1(y_t)
        y_hyper_t_p1 = self.temporal_proj_head_hyper_p1(y_hyper_t)

        return y_t_p0, y_hyper_t_p0, y_masked_t_p0, y_hyper_masked_t_p0, y_t_p1, y_hyper_t_p1, y_s_p0, y_hyper_s_p0, y_masked_s_p0, y_hyper_masked_s_p0, y_s_p1, y_hyper_s_p1, y_recon, y_recon_target, y_hyper_recon, y_hyper_recon_target,log_probs, log_probs_edge, torch.mean(torch.abs(x_ada_vis)), torch.mean(torch.abs(x_ada_vis_edge)), forecasted_value, forecasted_value_hyper

    @staticmethod
    def loss_nce(x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0

        return loss

    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src_lifted_att = node_score[edge_index[0]]
        dst_lifted_att = node_score[edge_index[1]]
        edge_score = src_lifted_att * dst_lifted_att
        return edge_score
