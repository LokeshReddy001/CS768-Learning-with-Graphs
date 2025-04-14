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

    
def run_SAGE(data, num_features, num_classes, setting="tran"):   
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask 
    if setting == "ind":
        test_mask = data.inductive_mask
    
    train_loader = NeighborLoader(
        data,
        input_nodes=train_mask,
        num_neighbors=[5, 5],
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
    
    
    model = SAGE(
        num_layers=2,
        input_dim=num_features,
        hidden_dim=128,
        output_dim=num_classes,
        dropout_ratio=0,
        activation=nn.functional.relu,
        norm_type="batch"
    )
    
    device = 'cuda'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 201):
        loss = train_sage(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_sage(model, val_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            test_acc =evaluate_sage(model, test_loader, device)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    mb_t, z_soft = model.forward(data.x.to(device), data.edge_index.to(device))
    return mb_t, z_soft

def run_model(model, features, edge_index, labels, train_mask, val_idx, test_idx):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 150):
        loss = train(model, features, edge_index, labels, train_mask, optimizer, criterion)
        val_acc = evaluate(model, features, edge_index, labels, val_idx)
        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate(model, features, edge_index, labels, test_idx)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")