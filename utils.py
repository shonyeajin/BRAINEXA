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
import dataset

# model
scalar = 20
eps = 1e-10

def DHT(edge_index, batch, add_loops=True):
    num_edge = edge_index.size(1)
    device = edge_index.device

    ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
    edge_to_node_index = torch.arange(0, num_edge, 1, device=device).repeat_interleave(2).view(1, -1)
    hyperedge_index = edge_index.T.reshape(1, -1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()

    ### Transform batch of nodes to batch of edges
    edge_batch = hyperedge_index[1, :].reshape(-1, 2)[:, 0]
    edge_batch = torch.index_select(batch, 0, edge_batch)

    ### Add self-loops to each node in the dual hypergraph
    if add_loops:
        bincount = hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]] != 1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([torch.arange(0, num_edge, 1, device=device).view(1, -1),
                           torch.arange(max_edge + 1, max_edge + num_edge + 1, 1, device=device).view(1, -1)],
                          dim=0)

        hyperedge_index = torch.cat([hyperedge_index[:, mask], loops], dim=1)

    return hyperedge_index, edge_batch



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def club(feature_p0, feature_p1, features_d):
    mu = feature_p0
    logvar = feature_p1
    batch_size = feature_p0.size(0)

    prediction_tile = mu.unsqueeze(1).expand(-1, batch_size, -1)
    feature_d_tile = features_d.unsqueeze(0).expand(batch_size, -1,-1)

    positive = -(mu-features_d)**2 / 2. / torch.exp(logvar)
    negative = -torch.mean((feature_d_tile - prediction_tile)**2, dim=1) / 2. / torch.exp(logvar)
    lld = torch.sum(positive, dim=-1)
    bound = torch.sum(positive, dim=-1) - torch.sum(negative, dim=-1)
    return lld, bound




