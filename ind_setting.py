import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np


def add_inductive_settings(data, spr=0.2):
    unlabeled_indices = torch.where(~data.train_mask)[0]

    num_unlabeled = len(unlabeled_indices)
    num_inductive = int(0.2 * num_unlabeled)
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

if __name__ == "__main__":

    dataset = Planetoid(root='/tmp/cora', name='Cora')
    data = dataset[0]
    data = add_inductive_settings(data)
    print(data.keys())

