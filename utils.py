import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon, Planetoid
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_args_with_config(args, config):
    merged = dict(config.get('global', {}))
    
    dataset_cfg = config.get(args.dataset, {})
    model_cfg = dataset_cfg.get(args.model, {})
    
    merged.update(model_cfg)
    
    merged['dataset'] = args.dataset
    merged['model'] = args.model
    
    return merged

def get_data(dataset_name, setting="transductive", seed=1234):
    torch.manual_seed(seed)
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./tmp/' + dataset_name.lower(), name=dataset_name)
        data = dataset[0]
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root='./tmp/' + dataset_name.lower(), name=dataset_name)
        data = dataset[0]
        data = create_masks(data)
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        name = dataset_name.split('-')[1]
        dataset = PygNodePropPredDataset(name=dataset_name, root='./tmp/' + name)
        data = dataset[0]
        data = create_masks(data)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    data.num_classes = dataset.num_classes

    if setting == "inductive":
        data = add_inductive_masks(data)

    return data

def create_masks(data):
    data.y = data.y.reshape(-1)
    labels = data.y
    num_classes = labels.max().item() + 1
    num_nodes = len(labels)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0))]  # Shuffle

        train_idx = idx[:20]
        val_idx = idx[20:50]
        test_idx = idx[50:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def add_inductive_masks(data, ind_split_ratio=0.2):
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]
    num_test = test_indices.size(0)
    perm = torch.randperm(num_test)
    split = int(ind_split_ratio * num_test)
    tran_indices = test_indices[perm[:split]]
    ind_indices = test_indices[perm[split:]]

    data.test_tran_mask = torch.zeros_like(data.test_mask)
    data.test_ind_mask = torch.zeros_like(data.test_mask)
    data.test_tran_mask[tran_indices] = True
    data.test_ind_mask[ind_indices] = True

    return data