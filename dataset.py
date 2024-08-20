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

class Mdd(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        print('dataset input part')





def get_split(args, fold=5):
    dataset = Mdd(root=data_dir + '/mutag')
    DS = args.dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    dataset = TUDataset(path, name=DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits

def sparsify_adjacency_matrix(adj_matrix, keep_percentage):
    threshold_value = np.percentile(adj_matrix, 100 - keep_percentage)
    sparse_adj_matrix = np.where(adj_matrix >= threshold_value, adj_matrix, 0.0)

    return sparse_adj_matrix



def get_random_split_idx(dataset, random_state=None, val_per=0.1, test_per=0.1, classification_mode=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    n_val = int(val_per * len(idx))
    n_test = int(test_per * len(idx))
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx_raw = idx[n_val + n_test:]
    normal_mask = (dataset.data.label[train_idx_raw] == 0).numpy() 
    if classification_mode:
        train_idx = train_idx_raw 
    else: 
        train_idx = train_idx_raw[normal_mask]

    ano_mask_test = (dataset.data.label[test_idx] == 1).numpy() 
    explain_idx = test_idx[ano_mask_test]

    return {'train': train_idx, 'val': val_idx,'test': test_idx, 'explain': explain_idx}

def get_loaders_mdd(batch_size, batch_size_test, dataset, split_idx=None):
    train_loader = DataLoader(dataset[split_idx['train']], batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx['val']], batch_size = batch_size_test, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size = batch_size_test, shuffle=False)
    explain_loader = DataLoader(dataset[split_idx['explain']], batch_size= 1, shuffle=False)
    return {'train':train_loader, 'val': val_loader,'test':test_loader, 'explain':explain_loader}

def get_data_loaders(dataset_name, batch_size, batch_size_test=None, random_state=0, data_dir='data'):
    dataset = Mdd(root=data_dir +'/mdd_aal')

    dataset.data.label = dataset.data.label.squeeze()
    dataset.data.label = 1 - dataset.data.label
    split_idx = get_random_split_idx(dataset, random_state)
    loaders = get_loaders_mdd(batch_size, batch_size_test, dataset=dataset, split_idx=split_idx)
    num_feat = dataset.data.signal.shape[2] 
    num_edge_feat = 0 
    num_node = dataset.data.signal.shape[1]
    num_time = dataset.data.signal.shape[0] // dataset.data.label.shape[0]

    
    meta = {'num_node': num_node,'num_feat': num_feat, 'num_edge_feat': num_edge_feat, 'num_time': num_time}

    return loaders, meta

    


def get_data_loaders_TU(args, split):
    DS = args.dataset

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    if DS in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=DS)

    dataset_num_features = dataset.num_node_features

    data_list = []
    label_list = []

    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())

    (train_index, test_index) = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]

    data_train = []
    for data in data_train_:
        if data.y != 0: 
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 0 else 0 

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'num_edge_feat':0}
    loader_dict = {'train': dataloader, 'test': dataloader_test}

    return loader_dict, meta
