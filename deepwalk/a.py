# %%
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T

# Load the CORA dataset
dataset = Planetoid(root='../Cora', name='Cora')

# Get the data object
data = dataset[0]
# %%
# Now data contains:
# data.x: Node features
# data.edge_index: Graph connectivity
# data.y: Node labels
# data.train_mask/val_mask/test_mask: Masks for splitting the dataset

print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of node features: {data.num_features}")
print(f"Number of classes: {dataset.num_classes}")

graph = data.edge_index
print(graph.shape)
# %%

adjacency_list = {}

for u, v in graph.t().tolist():
    if u not in adjacency_list:
        adjacency_list[u] = []
    adjacency_list[u].append(v)
    if v not in adjacency_list:
        adjacency_list[v] = []
    adjacency_list[v].append(u)
# %%






# %%
