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
import model

torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore")

# dataset name
dataset_name = ''

# argument
def arg_parse():
    parser = argparse.ArgumentParser(description='SIGNET')
    parser.add_argument('--dataset', type=str, default='mutag')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=9999)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--encoder_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max'])
    parser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    parser.add_argument('--explainer_model', type=str, default='gin', choices=['mlp', 'gin'])
    parser.add_argument('--explainer_layers', type=int, default=5)
    parser.add_argument('--explainer_hidden_dim', type=int, default=8)
    parser.add_argument('--explainer_readout', type=str, default='add', choices=['concat', 'add', 'last'])
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--decoder_layers', type=int, default=1)
    parser.add_argument('--alpha', type=float, default = 10.0)
    parser.add_argument('--temporal_kernel_size', type=int, default=5)
    parser.add_argument('--edge_importance_weighting', type=bool, default=False)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    return parser.parse_args()

# gpu check

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    temp = [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
    for i, itm in enumerate(temp):
        if itm['index']=='3':
            return itm



def visualize(G, cnt, color, epoch=None, loss=None):
    node_score = color[0].cpu()
    edge_score = color[1].cpu()

    plt.figure(figsize=(100,100))

    G1 = nx.Graph()
    G1.add_nodes_from([("node{}".format(n[0]), {'weight':node_score[i].item()}) for i, n in enumerate(G.nodes(data=True))])

    G1.add_edges_from(
        ("node{}".format(e[0]), "node{}".format(e[1]), {'weight':edge_score[i].item()}) for i, e in enumerate(G.edges(data=True))
    )

    pos = nx.spring_layout(G1)

    nx.draw_networkx_nodes(
        G1, pos, node_color=[n[1]['weight'] for n in G1.nodes(data=True)], node_shape='h',
        node_size=3000, cmap=plt.cm.Blues, alpha=0.9
                      )
    nx.draw_networkx_edges(
        G1, pos, edge_color=[e[2]['weight'] for e in G1.edges(data=True)],
        width=5, edge_cmap=plt.cm.Greys
    )
    nx.draw_networkx_labels(
        G1, pos, font_family='sans-serif', font_color='black', font_size=10, font_weight='bold'
    )

    # pos = nx.kamada_kawai_layout(G)
    # nx.draw_networkx_nodes(G,pos, node_color=node_score, cmap=plt.cm.Blues, alpha=0.9, node_size=2000)
    # print(f'node score max, min :{torch.max(node_score).item(), torch.min(node_score).item()}')
    # print(f'edge score max, min :{torch.max(edge_score).item(), torch.min(edge_score).item()}')
    # print(f'edge score shape: {edge_score.shape, node_score.shape}, edge_score type: {type(edge_score)}')
    # edge_score = torch.squeeze(edge_score)
    # print(f'edge_score:{edge_score.shape}')
    # edge_score = edge_score.numpy()

    # nx.draw_networkx_edges(G,pos, edge_color=edge_score, width = 5, edge_cmap=plt.cm.Blues, style='dashed')
    plt.show()
    plt.savefig('./img/img' + str(cnt) + '.png')
    plt.clf()




def run(args, device, seed, split=None):
    set_seed(seed)
    loaders, meta = get_data_loaders(args.dataset, args.batch_size, args.batch_size_test, random_state=seed)
    n_feat = meta['num_feat']
    n_edge_feat = meta['num_edge_feat']
    n_node = meta['num_node']
    n_time = meta['num_time']
    model = SIGNET(n_node, n_time, n_feat, n_edge_feat, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_rec_mean = nn.MSELoss()
    loss_rec_none = nn.MSELoss(reduction='none')

    explain_loader = loaders['explain']
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    best_model_name =''
    best_auc = 0
    
    epochs_vis = list(range(args.epochs))
    val_auc_vis = []
    test_auc_vis = []


    check_1s=[] # temp mi max
    check_2s=[] # temp mi min
    check_3s=[] # spa mi max
    check_4s=[] # spa mi min
    check_5s=[] # rec
    check_6s=[] # sampling
    check_7s=[] # lld

    bottleneck_count = torch.zeros(116).to(device)


    for epoch in range(1, args.epochs+1):
        
        model.train()
        loss_all = 0
        num_sample = 0


        temp_max=[]
        temp_min_view1=[]
        temp_min_view2=[]
        spa_max=[]
        spa_min_view1=[]
        spa_min_view2=[]
        rec_view1=[]
        rec_view2=[]
        fore_view1=[]
        fore_view2=[]
 

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            y_p0, y_hyper_p0, y_masked_p0, y_hyper_masked_p0, y_p1, y_hyper_p1, y_s_p0, y_hyper_s_p0, y_masked_s_p0, y_hyper_masked_s_p0, y_s_p1, y_hyper_s_p1, y_recon, y_recon_target, y_hyper_recon, y_hyper_recon_target, msk_log_prob, msk_log_prob_edge, x_norm, x_edge_norm, forecasted_value, forecasted_value_hyper = model(data, bottleneck_count)
            loss_rec_sample = torch.mean(loss_rec_none(y_recon_target, y_recon), dim=-1).flatten().unsqueeze(0).clone().detach()
            loss_rec_sample_hyper = torch.mean(loss_rec_none(y_hyper_recon_target, y_hyper_recon), dim=-1).flatten().unsqueeze(0).clone().detach()
            msk_log_prob = msk_log_prob.flatten().unsqueeze(-1)
            msk_log_prob_edge = msk_log_prob_edge.permute(1, 0, 2).contiguous().flatten().unsqueeze(-1)

            loss_sampling = -torch.mean(torch.mm(loss_rec_sample,msk_log_prob))
            loss_sampling_hyper = -torch.mean(torch.mm(loss_rec_sample_hyper, msk_log_prob_edge))

            # club bound calculate
            mi_temp_view1_lld, mi_temp_view1_bound = club(y_p0, y_p1, y_masked_p0)
            mi_temp_view2_lld, mi_temp_view2_bound = club(y_hyper_p0, y_hyper_p1, y_hyper_masked_p0)
            mi_spa_view1_lld, mi_spa_view1_bound = club(y_s_p0, y_s_p1, y_masked_s_p0)
            mi_spa_view2_lld, mi_spa_view2_bound = club(y_hyper_s_p0, y_hyper_s_p1,y_hyper_masked_s_p0)

            
            fore_batch, fore_time, fore_node, fore_dim = y_recon_target.size()
            forecasted_target = y_recon_target.permute(1,0,2,3).contiguous()

            forecasted_value = forecasted_value.view(fore_batch, fore_node, fore_time, fore_dim).permute(2,0,1,3).contiguous()

            forecasted_value = forecasted_value[:4,:,:,:]
            forecasted_target = forecasted_target[1:,:,:,:] 
            
            forecasted_value_hyper = forecasted_value_hyper.permute(1,0,2).contiguous()
            forecasted_value_hyper = forecasted_value_hyper[:4,:,:]
            forecasted_target_hyper = y_hyper_recon_target[1:,:,:]

            loss = model.loss_nce(y_masked_p0, y_hyper_masked_p0, args.temperature).mean() \
                + mi_temp_view1_bound.mean() \
                + mi_temp_view2_bound.mean() \
                + model.loss_nce(y_masked_s_p0, y_hyper_masked_s_p0, args.temperature).mean() \
                + mi_spa_view1_bound.mean() \
                + mi_spa_view2_bound.mean() \
                + 1e+2 * loss_rec_mean(y_recon_target, y_recon) + 1e+2 * loss_rec_mean(y_hyper_recon_target, y_hyper_recon)\
                + 1e+3 * loss_rec_mean(forecasted_target, forecasted_value) + 1e+3 * loss_rec_mean(forecasted_target_hyper, forecasted_value_hyper)\
                + 1e-3 * loss_sampling + 1e-4 * loss_sampling_hyper\
                - mi_temp_view1_lld.mean() - mi_temp_view2_lld.mean() - mi_spa_view1_lld.mean() -mi_spa_view2_lld.mean()
            

            temp_max.append(model.loss_nce(y_masked_p0, y_hyper_masked_p0, args.temperature))
            temp_min_view1.append(mi_temp_view1_bound)
            temp_min_view2.append(mi_temp_view2_bound)
            spa_max.append(model.loss_nce(y_masked_s_p0, y_hyper_masked_s_p0, args.temperature))
            spa_min_view1.append(mi_spa_view1_bound)
            spa_min_view2.append(mi_spa_view2_bound)
            rec_view1.extend(loss_rec_none(y_recon_target, y_recon).flatten().tolist())
            rec_view2.extend(loss_rec_none(y_hyper_recon_target, y_hyper_recon).flatten().tolist())
            fore_view1.extend(loss_rec_none(forecasted_target, forecasted_value).flatten().tolist())
            fore_view2.extend(loss_rec_none(forecasted_target_hyper, forecasted_value_hyper).flatten().tolist())

            loss_all += loss.item() * data.num_graphs
            num_sample += data.num_graphs
            loss.backward()
            optimizer.step()


            del data.signal
            del data.edge_index
            del data.batch
            del data
            gc.collect()
            torch.cuda.empty_cache()

        info_train = 'Epoch {:3d}, Loss CL {:.4f}'.format(epoch, loss_all / num_sample)

        temp_max_mu = torch.mean(torch.cat(temp_max))
        temp_max_sigma = torch.std(torch.cat(temp_max))
        temp_min_view1_mu = torch.mean(torch.cat(temp_min_view1))
        temp_min_view1_sigma = torch.std(torch.cat(temp_min_view1))
        temp_min_view2_mu = torch.mean(torch.cat(temp_min_view2))
        temp_min_view2_sigma = torch.std(torch.cat(temp_min_view2))
        spa_max_mu = torch.mean(torch.cat(spa_max))
        spa_max_sigma = torch.std(torch.cat(spa_max))
        spa_min_view1_mu = torch.mean(torch.cat(spa_min_view1))
        spa_min_view1_sigma = torch.std(torch.cat(spa_min_view1))
        spa_min_view2_mu = torch.mean(torch.cat(spa_min_view2))
        spa_min_view2_sigma = torch.std(torch.cat(spa_min_view2))
        rec_view1_mu = torch.mean(torch.tensor(rec_view1))
        rec_view1_sigma = torch.std(torch.tensor(rec_view1))
        rec_view2_mu = torch.mean(torch.tensor(rec_view2))
        rec_view2_sigma = torch.std(torch.tensor(rec_view2))
        fore_view1_mu = torch.mean(torch.tensor(fore_view1))
        fore_view1_sigma = torch.std(torch.tensor(fore_view1))
        fore_view2_mu = torch.mean(torch.tensor(fore_view2))
        fore_view2_sigma = torch.std(torch.tensor(fore_view2))

        if epoch % args.log_interval == 0:
            model.eval()
            # anomaly detection
            all_ad_true = []
            all_ad_score = []
            for data in val_loader:
                all_ad_true.append(data.label.cpu())
                ad_true_check =data.label.cpu()
                data = data.to(device)
                with torch.no_grad():
                    y_p0, y_hyper_p0, y_masked_p0, y_hyper_masked_p0, y_p1, y_hyper_p1, y_s_p0, y_hyper_s_p0, y_masked_s_p0, y_hyper_masked_s_p0, y_s_p1, y_hyper_s_p1, y_recon, y_recon_target, y_hyper_recon, y_hyper_recon_target, msk_log_prob, msk_log_prob_edge, _, _, forecasted_value, forecasted_value_hyper = model(data, bottleneck_count)
                    
                    mi_temp_view1_lld, mi_temp_view1_bound = club(y_p0, y_p1, y_masked_p0)
                    mi_temp_view2_lld, mi_temp_view2_bound = club(y_hyper_p0, y_hyper_p1, y_hyper_masked_p0)
                    mi_spa_view1_lld, mi_spa_view1_bound = club(y_s_p0, y_s_p1, y_masked_s_p0)
                    mi_spa_view2_lld, mi_spa_view2_bound = club(y_hyper_s_p0, y_hyper_s_p1,y_hyper_masked_s_p0)

                    batch_num = model.loss_nce(y_masked_p0, y_hyper_masked_p0, args.temperature).shape[0]

                    fore_batch, fore_time, fore_node, fore_dim = y_recon_target.size()
                    forecasted_target = y_recon_target.permute(1,0,2,3).contiguous()
                    forecasted_value = forecasted_value.view(fore_batch, fore_node, fore_time, fore_dim).permute(2,0,1,3).contiguous()
                    forecasted_value = forecasted_value[:4,:,:,:]
                    forecasted_target = forecasted_target[1:,:,:,:] 
                    forecasted_value_hyper = forecasted_value_hyper.permute(1,0,2).contiguous()
                    forecasted_value_hyper = forecasted_value_hyper[:4,:,:]
                    forecasted_target_hyper = y_hyper_recon_target[1:,:,:]
                    f_t, _, f_n = forecasted_value_hyper.size()
                    forecasted_value_hyper = forecasted_value_hyper.view(f_t, batch_num, -1, f_n)
                    forecasted_target_hyper = forecasted_target_hyper.view(f_t, batch_num, -1, f_n)
    
                    temp_max_likelihood = stats.norm.pdf(model.loss_nce(y_masked_p0, y_hyper_masked_p0, args.temperature).cpu(), temp_max_mu.cpu(), temp_max_sigma.cpu())
                    temp_min_view1_likelihood = stats.norm.pdf(mi_temp_view1_bound.cpu(), temp_min_view1_mu.cpu(), temp_min_view1_sigma.cpu())
                    temp_min_view2_likelihood = stats.norm.pdf(mi_temp_view2_bound.cpu(), temp_min_view2_mu.cpu(), temp_min_view2_sigma.cpu())
                    

                    spa_max_likelihood = stats.norm.pdf(torch.mean(model.loss_nce(y_masked_s_p0, y_hyper_masked_s_p0, args.temperature).view(batch_num, -1), dim = 1).cpu(), spa_max_mu.cpu(), spa_max_sigma.cpu())
                    spa_min_view1_likelihood = stats.norm.pdf(torch.mean(mi_spa_view1_bound.view(batch_num,-1), dim=1).cpu(), spa_min_view1_mu.cpu(), spa_min_view1_sigma.cpu())
                    spa_min_view2_likelihood = stats.norm.pdf(torch.mean(mi_spa_view2_bound.view(batch_num, -1), dim=1).cpu(), spa_min_view2_mu.cpu(), spa_min_view2_sigma.cpu())
                    rec_view1_likelihood = stats.norm.pdf(torch.mean(loss_rec_none(y_recon_target, y_recon), dim=(1,2,3)).cpu(), rec_view1_mu.cpu(), rec_view1_sigma.cpu())
                    rec_view2_likelihood = stats.norm.pdf(torch.mean(loss_rec_none(y_hyper_recon_target, y_hyper_recon).view(n_time, batch_num, -1, n_feat), dim=(0,2,3)).cpu(), rec_view2_mu.cpu(), rec_view2_sigma.cpu())
                    fore_view1_likelihood = stats.norm.pdf(torch.mean(loss_rec_none(forecasted_target, forecasted_value), dim=(0,2,3)).cpu(), fore_view1_mu.cpu(), fore_view1_sigma.cpu())
                    fore_view2_likelihood = stats.norm.pdf(torch.mean(loss_rec_none(forecasted_target_hyper, forecasted_value_hyper), dim=(0,2,3)).cpu(), fore_view2_mu.cpu(), fore_view2_sigma.cpu())


                    ano_score= temp_max_likelihood\
                            +temp_min_view1_likelihood\
                            +temp_min_view2_likelihood\
                            +spa_max_likelihood\
                            +spa_min_view1_likelihood\
                            +spa_min_view2_likelihood\
                            +rec_view1_likelihood\
                            +rec_view2_likelihood\
                            +fore_view1_likelihood\
                            +fore_view2_likelihood

                    loss_rec_view1 = torch.mean(loss_rec_none(y_recon_target, y_recon), dim=(2,3))
                  
                all_ad_score.append(torch.tensor(ano_score))

                del data.signal
                del data.edge_index
                del data.batch
                del data
                gc.collect()
                torch.cuda.empty_cache()
            ad_true = torch.cat(all_ad_true)
            ad_score = torch.cat(all_ad_score)
            ad_auc_val = roc_auc_score(ad_true, ad_score)
            # to select optimal thresholding value
            fpr, tpr, thresholds = roc_curve(ad_true, ad_score)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            ad_pred = torch.where(ad_score>=optimal_threshold, 1., 0.)
            ad_accuracy_val = accuracy_score(ad_true, ad_pred)
            ad_f1_val = f1_score(ad_true, ad_pred)
            ad_recall_val = recall_score(ad_true, ad_pred)
            ad_precision_val = precision_score(ad_true, ad_pred)
            tn, fp, fn, tp = confusion_matrix(ad_true, ad_pred).ravel()
            ad_specificity_val = tn / (tn + fp)


            abn_loss= []
            n_loss = []
            for i, itm in enumerate(ad_true):
                if itm == 1: 
                    abn_loss.append(ad_score[i].item())
                else:
                    n_loss.append(ad_score[i].item())


            info_val = 'AD_AUC_VAL:{:.4f}'.format(ad_auc_val)
            info_vals = '[VAL] ACC:{:.4f}, '.format(ad_accuracy_val)\
                + 'F1:{:.4f}, '.format(ad_f1_val)\
                + 'RECALL:{:.4f}, '.format(ad_recall_val)\
                + 'PRE:{:.4f}, '.format(ad_precision_val)\
                + 'SPE:{:.4f}'.format(ad_specificity_val)

            val_auc_vis.append(ad_auc_val)
    return ad_auc_val, best_auc

# main
if __name__ == '__main__':
    args = arg_parse()

    # device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    print('Current cuda device: ', torch.cuda.current_device()) 
    print('Count of using GPUs: ', torch.cuda.device_count())

    ad_aucs = []
    key_auc_list = []
    splits=[None]*args.num_trials
    best_auc = 0
    for trial in range(args.num_trials):
        results, best_temp = run(args, device, seed=trial, split=splits[trial])