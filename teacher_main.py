import argparse
import torch
from torch_geometric.datasets import Planetoid

from teacher_models import SAGE, GCN, GAT, APPNP
from teacher_run import train_sage, evaluate_sage, run_SAGE, train, evaluate, run_model

dataset = Planetoid(root='./Cora', name='Cora')

data = dataset[0]
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

def add_inductive_settings(data, spr=0.2):
    unlabeled_indices = torch.where(data.test_mask)[0]

    num_unlabeled = len(unlabeled_indices)
    num_inductive = int(spr * num_unlabeled)
    perm = torch.randperm(num_unlabeled)
    inductive_indices = unlabeled_indices[perm[:num_inductive]]
    observed_indices = unlabeled_indices[perm[num_inductive:]]

    data.observed_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.inductive_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.observed_mask[observed_indices] = True
    data.inductive_mask[inductive_indices] = True

    edge_index = data.edge_index
    src, dst = edge_index

    mask = ~data.inductive_mask[src] & ~data.inductive_mask[dst]
    data.ind_edge_index = edge_index[:, mask]

    return data

def load_data(dataset):
    if dataset == "cora":
        dataset = Planetoid(root='./Cora', name='Cora')
        data = add_inductive_settings(dataset[0])
        
        features = data.x
        labels = data.y
        
        return data.edge_index, features, labels, data.train_mask, data.val_mask, data.test_mask, data.ind_edge_index, data.observed_mask, data.inductive_mask

edge_index, features, labels, train_mask, val_idx, test_idx, ind_edge_index, obs_mask, ind_mask  = load_data("cora")
# edge_index, features, labels, train_mask, val_idx, test_idx  = load_data("cora")

data = add_inductive_settings(data, 0.2)


