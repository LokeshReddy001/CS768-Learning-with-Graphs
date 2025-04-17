import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from models import SAGE, GCN, APPNP_Model, GAT
from utils import get_data, load_config
from argparse import Namespace, ArgumentParser

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

def run_SAGE(args, data, mode="transductive"):   
    num_classes = data.num_classes
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    if mode == "inductive":
        test_mask = data.test_ind_mask

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
    
    #TODO: don't know what's val mask in ind mode
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
    if mode == "inductive":
        test_obs_loader = NeighborLoader(
            data,
            input_nodes=data.test_tran_mask,
            num_neighbors=[-1, -1],
            batch_size=32
        )
    
    
    model = SAGE(
        num_layers=args.num_layers,
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        dropout_ratio=args.dropout_ratio,
        activation=nn.functional.relu,
        norm_type="batch"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    patience = args.patience if hasattr(args, 'patience') else 10
    patience_counter = 0
    best_val_acc = 0
    best_model_state = None

    for epoch in range(1, 101):
        loss = train_sage(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == 1 or epoch == 100:
            val_acc = evaluate_sage(model, val_loader, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | No improvement, patience: {patience_counter}/{patience}")
    
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    if mode=="inductive":
        test_acc = evaluate_sage(model, test_loader, device)
        test_obs_acc = evaluate_sage(model, test_obs_loader, device)
        prod_acc = 0.2*test_acc+0.8*test_obs_acc

        print(f"Test_ind Acc: {test_acc:.4f} | Test_tran Acc: {test_obs_acc:.4f} | Test_prod Acc: {prod_acc:.4f}")
    else:
        test_acc = evaluate_sage(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.4f}")
    

    emb_t, z_soft = model.forward(data.x.to(device), data.edge_index.to(device))
    return emb_t, z_soft


def run_model(args, model, data, mode="transductive"):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Early stopping parameters
    patience = args.patience if hasattr(args, 'patience') else 10
    patience_counter = 0
    best_val_acc = 0
    best_model_state = None
    
    max_epochs = 200
    
    if mode=="inductive":
        for epoch in range(1, max_epochs + 1):
            loss = train(model, data.x, data.ind_edge_index, data.y, data.train_mask, optimizer, criterion)
            
            if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
                val_acc = evaluate(model, data.x, data.edge_index, data.y, data.val_mask)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                    patience_counter = 0
                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | No improvement, patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch}")
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        test_tran_acc = evaluate(model, data.x, data.edge_index, data.y, data.test_tran_mask)
        test_ind_acc = evaluate(model, data.x, data.edge_index, data.y, data.test_ind_mask)
        prod_acc = 0.8*test_tran_acc+0.2*test_ind_acc
        print(f"Final Test_ind Acc: {test_ind_acc:.4f} | Test_tran Acc: {test_tran_acc:.4f} | prod Acc: {prod_acc:.4f}")
    
    else:
        for epoch in range(1, max_epochs + 1):
            loss = train(model, data.x, data.edge_index, data.y, data.train_mask, optimizer, criterion)
            
            if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
                val_acc = evaluate(model, data.x, data.edge_index, data.y, data.val_mask)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                    patience_counter = 0
                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | No improvement, patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch}")
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        test_acc = evaluate(model, data.x, data.edge_index, data.y, data.test_mask)
        print(f"Final Test Accuracy: {test_acc:.4f}")
    
    emb_t, z_soft = model.forward(data.x.to(device), data.edge_index.to(device))
    return emb_t, z_soft

def run_GCN(args, data, mode="transductive"):
    gcn_model = GCN(
        num_layers=args.num_layers,
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=data.num_classes,
        dropout_ratio=args.dropout_ratio,
        activation=nn.functional.relu,
        norm_type="batch"
    )
    return run_model(args, gcn_model, data, mode)


def run_APPNP(args, data, mode="transductive"):
    appnp_model = APPNP_Model(
        num_layers=args.num_layers,
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=data.num_classes,
        dropout_ratio=args.dropout_ratio,
        activation=F.relu,
    )
    return run_model(args, appnp_model, data, mode)

def run_GAT(args, data, mode="transductive"):
    gat_model = GAT(
        num_layers=args.num_layers,
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=data.num_classes,
        dropout_ratio=args.dropout_ratio,
        activation=F.relu,
        num_heads=args.num_heads,
        attn_drop=args.attn_dropout_ratio,
        negative_slope=0.2,
        residual=True
    )
    
    return run_model(args, gat_model, data, mode)

def main(args):
    model_name, dataset_name, mode = args.teacher, args.dataset, args.mode
    
    data = get_data(dataset_name, mode, args.seed)
    
    torch.manual_seed(args.seed)

    if mode == "inductive":
        edge_index = data.edge_index
        src, dst = edge_index
        mask = ~data.test_ind_mask[src] & ~data.test_ind_mask[dst]
        data.ind_edge_index = edge_index[:, mask]
    
    if model_name == "SAGE":
        emb_t, z_t = run_SAGE(args, data, mode)
    elif model_name == "GCN":
        emb_t, z_t = run_GCN(args, data, mode)
    elif model_name == "APPNP":
        emb_t, z_t = run_APPNP(args, data, mode)
    elif model_name == "GAT":
        emb_t, z_t = run_GAT(args, data, mode)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    output_dir = "./teacher_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(emb_t, os.path.join(output_dir, f"dataset_{dataset_name}_model_{model_name}_mode_{args.mode}_seed_{args.seed}_embeddings.pt"))
    torch.save(z_t, os.path.join(output_dir, f"dataset_{dataset_name}_model_{model_name}_mode_{args.mode}_seed_{args.seed}_logits.pt"))
    print(f"Embeddings and softmax outputs saved to {output_dir}")
    print("Training completed.")

if __name__ == "__main__":
    parser = ArgumentParser(description='Graph Neural Network Teacher Model Training')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--teacher', type=str, default='SAGE', help='Model name')
    parser.add_argument('--config', type=str, default='tran.conf.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='transductive', help='mode: transductive or inductive')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()
    
    config = load_config(args.config)

    global_args = config['global']
    model_args = config[args.dataset][args.teacher]

    final_args = {**global_args, **model_args, **vars(args)}
    
    if 'learning_rate' not in final_args.keys():
        final_args['learning_rate'] = 0.01

    main(Namespace(**final_args))