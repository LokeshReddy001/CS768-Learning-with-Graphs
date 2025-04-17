import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

def train_sage(model, loader, optimizer, criterion, device, homo=True):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        x = batch.x
        y = batch.y[:batch.batch_size]  # Only use input nodes

        if homo:
            edge_index = batch.edge_index
        else:
            rel = list(batch.edge_index_dict.keys())[0]
            edge_index = batch.edge_index_dict[rel]

        _, out = model(x, edge_index)
        out = out[:batch.batch_size]  # Only use predictions for input nodes

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_sage(model, loader, device, homo=True):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        y = batch.y[:batch.batch_size]  # Only input nodes

        if homo:
            edge_index = batch.edge_index
        else:
            rel = list(batch.edge_index_dict.keys())[0]
            edge_index = batch.edge_index_dict[rel]

        _, out = model(x, edge_index)
        out = out[:batch.batch_size]  # Only predictions for input nodes

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total

def train(model, data, edge_index, labels, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _, out = model(data, edge_index)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, edge_index, labels, idx):
    model.eval()
    _, out = model(data, edge_index)
    pred = out[idx].argmax(dim=1)
    correct = (pred == labels[idx]).sum().item()
    acc = correct / sum(idx)
    return acc
    
# save embeddings, softmax scores tensors above in a directory
def save_tensors(emb_t, z_soft, output_dir):
    torch.save(emb_t, f"{output_dir}/embeddings.pt")
    torch.save(z_soft, f"{output_dir}/label_scores.pt")
# Example usage
output_dir = "./teacher_outputs"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_SAGE(args, data, num_features, num_classes, setting="tran"):   
    train_mask, val_mask, test_mask, obs_mask = data.train_mask, data.val_mask, data.test_mask, data.observed_mask 
    if setting == "ind":
        test_mask = data.inductive_mask

    fan_out = []
    for i in args.fan_out.split(","):
        fan_out.append(int(i))
    
    train_loader = NeighborLoader(
        data,
        input_nodes=train_mask,
        num_neighbors=fan_out,
        batch_size=32,
        shuffle=True
    )
    
    #TODO: don't know what's val mask in ind setting
    val_loader = NeighborLoader(
        data,
        input_nodes=val_mask,
        num_neighbors=[-1, -1],
        batch_size=32
    )
    
    test_loader = NeighborLoader(
        data,
        input_nodes=test_mask,
        num_neighbors=[-1, -1],
        batch_size=32
    )

    test_obs_loader = []
    if setting == "ind":
        test_obs_loader = NeighborLoader(
            data,
            input_nodes=obs_mask,
            num_neighbors=[-1, -1],
            batch_size=32
        )
    
    
    model = SAGE(
        num_layers=args.num_layers,
        input_dim=num_features,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        dropout_ratio=args.dropout_ratio,
        activation=nn.functional.relu,
        norm_type="batch"
    )
    
    device = 'cuda'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 101):
        loss = train_sage(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_sage(model, val_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate_sage(model, test_loader, device)
            if setting=="ind":
                test_obs_acc = evaluate_sage(model, test_obs_loader, device)
                prod_acc = 0.8*test_acc+0.2*test_obs_acc
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test_ind Acc: {test_acc:.4f} | Test_tran Acc: {test_obs_acc:.4f} | prod Acc: {prod_acc:.4f}")
            else:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test_obs Acc: {test_acc:.4f}")

    mb_t, z_soft = model.forward(data.x.to(device), data.edge_index.to(device))
    return mb_t, z_soft

def run_model(args, model, features, edge_index, labels, train_mask, val_mask, test_mask, setting="tran", orig_edge_index=[]):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    edge_index_eval = edge_index
    if setting=="ind":
        edge_index_eval = orig_edge_index
    
    for epoch in range(1, 200):
        loss = train(model, features, edge_index, labels, train_mask, optimizer, criterion)
        val_acc = evaluate(model, features, edge_index_eval, labels, val_mask)
        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate(model, features, edge_index_eval, labels, test_mask)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

def run_GCN(args, num_features, num_classes, features, edge_index, labels, train_mask, val_mask, test_mask, setting="tran",  orig_edge_index=[]):
    gcn_model = GCN(
        num_layers=args.num_layers,
        input_dim=num_features,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        dropout_ratio=args.dropout_ratio,
        activation=nn.functional.relu,
        norm_type="batch"
    )
    run_model(args, gcn_model, features, edge_index, labels, train_mask, val_mask, test_mask,  setting, orig_edge_index)

def run_APPNP(args, num_features, num_classes, features, edge_index, labels, train_mask, val_mask, test_mask,  setting="tran", orig_edge_index=[]):
    appnp_model = APPNP_Model(
        num_layers=args.num_layers,
        input_dim=num_features,  
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,  
        dropout_ratio=args.dropout_ratio,
        activation=F.relu,
    )
    
    run_model(args, appnp_model, features, edge_index, labels, train_mask, val_mask, test_mask, setting, orig_edge_index)

def run_GAT(args, num_features, num_classes, features, edge_index, labels, train_mask, val_mask, test_mask, setting="tran", orig_edge_index=[]):
    gat_model = GAT(
        num_layers=args.num_layers,
        input_dim=num_features,  
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,       
        dropout_ratio=args.dropout_ratio,
        activation=F.relu,
        num_heads=args.num_heads,
        attn_drop=args.attn_dropout_ratio,
        negative_slope=0.2,
        residual=True
    )
    
    run_model(args, gat_model, features, edge_index, labels, train_mask, val_mask, test_mask, setting, orig_edge_index)

def run(args, model_name, dataset_name, setting="tran"):
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:  
        dataset = Planetoid(root=os.path.join('/tmp', dataset_name.lower()), name=dataset_name)
        if model_name=="SAGE":
            data = dataset[0]
            data.ind_edge_index = []
            data.observed_mask = []
            data.inductive_mask = []
            test_mask = data.test_mask
            if setting=="ind":
                data = add_inductive_settings(data)
                test_mask = data.inductive_mask
            run_SAGE(args, data, dataset.num_node_features, dataset.num_classes, setting)
        else:
            num_features, num_classes, features, labels, edge_index, ind_edge_index, train_mask, val_mask, test_mask = load_data(dataset, setting)
            edges = edge_index
            if setting == "ind":
                edges = ind_edge_index
            if model_name == "GAT":
                run_GAT(args, num_features, num_classes, features, edges, labels, train_mask, val_mask, test_mask, setting, edge_index)
            elif model_name == "GCN":
                run_GCN(args, num_features, num_classes, features, edges, labels, train_mask, val_mask, test_mask, setting, edge_index)
            elif model_name == "APPNP":
                run_APPNP(args, num_features, num_classes, features, edges, labels, train_mask, val_mask, test_mask, setting, edge_index)
