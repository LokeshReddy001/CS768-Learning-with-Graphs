import torch
import torch.nn as nn
import torch.nn.functional as F
from deepwalk.deepwalk import DeepWalk
from models import MLP
import networkx as nx
from torch_geometric.datasets import Planetoid
import argparse
from utils import get_data
import yaml
import os

def pgd_delta(model, feats, labels, train_mask, eps=0.05, iters=5):
    alpha = eps / 4

    delta = torch.rand(feats.shape) * eps * 2 - eps
    delta = delta.to(feats.device)
    delta = torch.nn.Parameter(delta)

    for i in range(iters):
        p_feats = feats + delta

        _, logits = model(p_feats)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)

    output = delta.detach()
    return output

def train(mlp, optimizer, data, position_embeddings, teacher_emb, teacher_out, mode, 
          gt_weight, sl_weight, rsd_weight, adv_weight, temperature, device, trans_nodes_mask):
    mlp.train()
    optimizer.zero_grad()
    
    inp = torch.cat([data.x, position_embeddings.to(device)], dim=-1)  
    mlp_emb, mlp_out = mlp(inp)
    
    # GROUND TRUTH Cross Entropy Loss
    GT_loss = F.cross_entropy(mlp_out[data.train_mask], data.y[data.train_mask])
    
    # Set up loss components based on mode
    if mode == "inductive":
        trans_nodes = trans_nodes_mask.nonzero(as_tuple=True)[0]
        
        # SOFT LABELS KL Divergence Loss
        SL_Loss = F.kl_div(
            F.log_softmax(mlp_out[trans_nodes]/temperature, dim=1),
            F.softmax(teacher_out[trans_nodes]/temperature, dim=1),
            reduction='batchmean'
        ) * (temperature**2)
        
        # Representational Similarity Distillation Loss
        teacher_mat = teacher_emb[trans_nodes] @ teacher_emb[trans_nodes].t()
        encoded_mlp_meb = mlp.MLP_RSD(mlp_emb[trans_nodes])
        mlp_mat = encoded_mlp_meb @ encoded_mlp_meb.t()
        RSD_Loss = torch.mean((mlp_mat - teacher_mat) ** 2)
        
        # Adversarial Feature Augmentation Loss
        delta = pgd_delta(mlp, inp, data.y, data.train_mask)
        _, adv_mlp_out = mlp(inp + delta)
        ADV_loss = F.cross_entropy(adv_mlp_out[data.train_mask], data.y[data.train_mask]) + \
                  F.cross_entropy(adv_mlp_out[trans_nodes_mask], F.softmax(teacher_out[trans_nodes_mask], dim=1))
    else:
        # SOFT LABELS KL Divergence Loss
        SL_Loss = F.kl_div(
            F.log_softmax(mlp_out/temperature, dim=1),
            F.softmax(teacher_out/temperature, dim=1),
            reduction='batchmean'
        ) * (temperature**2)
        
        # Representational Similarity Distillation Loss
        teacher_mat = teacher_emb @ teacher_emb.t()
        encoded_mlp_meb = mlp.MLP_RSD(mlp_emb)
        mlp_mat = encoded_mlp_meb @ encoded_mlp_meb.t()
        RSD_Loss = torch.mean((mlp_mat - teacher_mat) ** 2)
        
        # Adversarial Feature Augmentation Loss
        delta = pgd_delta(mlp, inp, data.y, data.train_mask)
        _, adv_mlp_out = mlp(inp + delta)
        ADV_loss = F.cross_entropy(adv_mlp_out[data.train_mask], data.y[data.train_mask]) + \
                  F.cross_entropy(adv_mlp_out, F.softmax(teacher_out, dim=1))
    
    # Total loss with weighted components
    loss = gt_weight * GT_loss + sl_weight * SL_Loss + rsd_weight * RSD_Loss + adv_weight * ADV_loss
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test(mlp, data, position_embeddings, device, mode):
    mlp.eval()
    inp = torch.cat([data.x, position_embeddings.to(device)], dim=-1)
    _, out = mlp(inp)
    pred = out.argmax(dim=1)

    accs = []
    if mode == 'inductive':
        for mask in [data.train_mask, data.val_mask, data.test_tran_mask, data.test_ind_mask]:
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum()) / int(mask.sum())
            accs.append(acc)
        return accs  # train_acc, val_acc, test_tran_acc, test_ind_acc
    else:
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum()) / int(mask.sum())
            accs.append(acc)
        return accs

def main():
    parser = argparse.ArgumentParser(description='Graph Neural Network Student Model Training')
    
    # Dataset and mode arguments
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'Pubmed'],
                        help='Dataset name (default: Cora)')
    parser.add_argument('--mode', type=str, default='inductive', choices=['inductive', 'transductive'],
                        help='Training mode (default: inductive)')
    parser.add_argument('--teacher', type=str, default='SAGE', choices=['SAGE', 'GCN', 'GAT', 'APPNP'],
                        help='Teacher model type (default: SAGE)')
    # Model configuration
    parser.add_argument('--num_layers', type=int, default=3, help='Number of MLP layers (default: 3)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size (default: 64)')
    
    # DeepWalk configuration
    parser.add_argument('--walk_length', type=int, default=80, help='Length of random walks (default: 80)')
    parser.add_argument('--walks_per_vertex', type=int, default=10,
                        help='Number of walks per vertex (default: 10)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default: 1234)')
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs (default: 5)')
    
    # Teacher model paths
    parser.add_argument('--teacher_outs', type=str, default='teacher_outputs',
                        help='Path to base folder of teacher outputs (default: teacher_outputs)')
    
    # Loss weights
    parser.add_argument('--gt_weight', type=float, default=1.0, 
                        help='Weight for ground truth loss (default: 1.0)')
    parser.add_argument('--sl_weight', type=float, default=0.5, 
                        help='Weight for soft label loss (default: 0.5)')
    parser.add_argument('--rsd_weight', type=float, default=0.1, 
                        help='Weight for representational similarity loss (default: 0.0)')
    parser.add_argument('--adv_weight', type=float, default=0.3, 
                        help='Weight for adversarial loss (default: 0.5)')
    
    # Adversarial training parameters
    parser.add_argument('--pgd_eps', type=float, default=0.05, 
                        help='Epsilon for PGD perturbation (default: 0.05)')
    parser.add_argument('--pgd_iters', type=int, default=5, 
                        help='Number of PGD iterations (default: 5)')
    
    # Distillation parameters
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for soft labels (default: 1.0)')
    
    # Device
    parser.add_argument('--device', type=str, default=None, 
                        choices=['cpu', 'cuda'], help='Device to run on (default: auto)')
    
    # Test split ratio for inductive mode
    parser.add_argument('--test_split_ratio', type=float, default=0.8, 
                        help='Ratio for splitting test set in inductive mode (default: 0.8)')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Running in {args.mode} mode on {args.dataset} dataset")
    
    # Get dataset
    data = get_data(args.dataset, setting=args.mode, seed=args.seed)
    torch.manual_seed(args.seed)

    # Load teacher outputs
    teacher_emb_path = os.path.join(args.teacher_outs, f"dataset_{args.dataset}_model_{args.teacher}_mode_{args.mode}_seed_{args.seed}_embeddings.pt")
    teacher_logits_path = os.path.join(args.teacher_outs, f"dataset_{args.dataset}_model_{args.teacher}_mode_{args.mode}_seed_{args.seed}_logits.pt")

    try:
        teacher_emb = torch.load(teacher_emb_path, map_location=device)
        teacher_out = torch.load(teacher_logits_path, map_location=device)
    except FileNotFoundError:
        print(f"Teacher outputs not found at {teacher_emb_path} or {teacher_logits_path}. Exiting.")
        return

    if args.mode == 'inductive':
        trans_nodes_mask = ~data.test_ind_mask
    else:
        trans_nodes_mask = None
    
    # DeepWalk embeddings
    print("Generating DeepWalk embeddings...")
    graph_nx = nx.Graph()
    graph_nx.add_edges_from(data.edge_index.t().tolist())
    deepwalk_model = DeepWalk(graph_nx, 
                             walk_length=args.walk_length, 
                             walks_per_vertex=args.walks_per_vertex)
    deepwalk_model.train()
    position_embeddings = deepwalk_model.get_embeddings()
    print(f"DeepWalk embeddings shape: {position_embeddings.shape}")

    # Initialize MLP model
    mlp = MLP(num_layers=args.num_layers,
             input_feat_dim=data.x.shape[1],
             position_emb_dim=position_embeddings.shape[1],
             hidden_dim=args.hidden_dim,
             output_dim=data.num_classes)
    
    mlp = mlp.to(device)
    data = data.to(device)
    position_embeddings = torch.tensor(position_embeddings).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    patience = args.patience
    patience_counter = 0
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        loss = train(mlp, optimizer, data, position_embeddings, teacher_emb, teacher_out, 
                   args.mode, args.gt_weight, args.sl_weight, args.rsd_weight, args.adv_weight,
                   args.temperature, device, trans_nodes_mask)
        
        # Evaluate every args.eval_every epochs or on first and last epoch
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            if args.mode == 'inductive':
                train_acc, val_acc, test_tran_acc, test_ind_acc = test(mlp, data, position_embeddings, device, args.mode)
                
                # Store best model state based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {key: value.cpu().clone() for key, value in mlp.state_dict().items()}
                    best_epoch = epoch
                    patience_counter = 0
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (BEST), '
                          f'Test Tran Acc: {test_tran_acc:.4f}, Test Ind Acc: {test_ind_acc:.4f}')
                else:
                    patience_counter += 1
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                          f'Test Tran Acc: {test_tran_acc:.4f}, Test Ind Acc: {test_ind_acc:.4f}, '
                          f'No improvement, patience: {patience_counter}/{patience}')
            else:
                train_acc, val_acc, test_acc = test(mlp, data, position_embeddings, device, args.mode)
                
                # Store best model state based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {key: value.cpu().clone() for key, value in mlp.state_dict().items()}
                    best_epoch = epoch
                    patience_counter = 0
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (BEST), '
                          f'Test Acc: {test_acc:.4f}')
                else:
                    patience_counter += 1
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                          f'Test Acc: {test_acc:.4f}, '
                          f'No improvement, patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}")
                break
    
    # Load best model state for final evaluation
    if best_model_state is not None:
        mlp.load_state_dict(best_model_state)
        print(f"\nLoaded best model from epoch {best_epoch} with validation accuracy: {best_val_acc:.4f}")
    
    # Final evaluation with best model
    print('\nFinal results:')
    if args.mode == 'inductive':
        train_acc, val_acc, test_tran_acc, test_ind_acc = test(mlp, data, position_embeddings, device, args.mode)
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Test Tran Acc: {test_tran_acc:.4f}, Test Ind Acc: {test_ind_acc:.4f}')
    else:
        train_acc, val_acc, test_acc = test(mlp, data, position_embeddings, device, args.mode)
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    

if __name__ == '__main__':
    main()