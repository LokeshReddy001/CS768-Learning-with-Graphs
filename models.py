import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP

class MLP(nn.Module):
    def __init__(self, num_layers, input_feat_dim, position_emb_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.input_feat_dim = input_feat_dim
        self.position_emb_dim = position_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=0.5)
        self.layers = nn.ModuleList()
        self.rsd_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.layers.append(nn.Linear(input_feat_dim + position_emb_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, inp):
        for layer in self.layers[:-1]:
            inp = F.relu(layer(inp))
            inp = self.dropout(inp)
        out = self.layers[-1](inp)
        return inp, out
    
    def MLP_RSD(self, mlp_emb):
        return F.relu(self.rsd_encoder(mlp_emb))

class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(SAGEConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        h = x
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_list[-1], h


    def inference(self, x_all, edge_index, batch_size=1024, device="cuda"):
        """
        Full-graph inference using mini-batches (for large graphs).
        """
        from torch_geometric.loader import NeighborLoader

        x = x_all.to(device)
        for l, layer in enumerate(self.layers):
            new_x = torch.zeros(
                x_all.size(0),
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)

            loader = NeighborLoader(
                data=(x_all, edge_index),
                input_nodes=torch.arange(x_all.size(0)),
                num_neighbors=[-1],  # full neighbors
                batch_size=batch_size,
                shuffle=False
            )

            for batch in loader:
                batch = batch.to(device)
                h = x[batch.n_id]
                h = layer(h, batch.edge_index)

                if l != self.num_layers - 1:
                    if self.norm_type != "none":
                        h = self.norms[l](h)
                    h = self.activation(h)
                    h = self.dropout(h)

                new_x[batch.n_id[:batch.batch_size]] = h

            x = new_x
        return x
# For small, medium datasets few thousands, use model() in eval
# For large ones like 100k or millions, use inference

class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.activation = activation

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GCNConv(input_dim, output_dim))
        else:
            self.layers.append(GCNConv(input_dim, hidden_dim))
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
                if norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        h_list = []
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
                h_list.append(h)
        return h_list[-1], h

class GAT(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation=F.relu,
            num_heads=8,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=False,
    ):
        super().__init__()
        
        assert num_layers > 1

        hidden_dim //= num_heads 
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)

        heads = [num_heads] * (num_layers - 1) + [1]  
        # heads = ([num_heads] * num_layers) + [1]

        # Input layer
        self.layers.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads[0],
                dropout=attn_drop,
                negative_slope=negative_slope,
                concat=True, 
            )
        )

        # Hidden layers
        for l in range(1, num_layers - 1):
            self.layers.append(
                GATConv(
                    in_channels=hidden_dim * heads[l - 1],  
                    out_channels=hidden_dim,
                    heads=heads[l],
                    dropout=attn_drop,
                    negative_slope=negative_slope,
                    concat=True, 
                )
            )

        # Output layer
        self.layers.append(
            GATConv(
                in_channels=hidden_dim * heads[-2],  
                out_channels=output_dim,
                heads=heads[-1],  
                dropout=attn_drop,
                negative_slope=negative_slope,
                concat=False, 
            )
        )

    def forward(self, x, edge_index):
        h_list = []
        h = x
        for l, layer in enumerate(self.layers):
            h = self.dropout(h) 
            h = layer(h, edge_index)
            if l != self.num_layers - 1:
                h = self.activation(h)  
                h_list.append(h)
        return h_list[-1], h
    
class APPNP_Model(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation=F.relu,
            norm_type="none",
            edge_drop=0,
            alpha=0.1,
            k=10,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            # Output layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNP(K=k, alpha=alpha, dropout=edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x, edge_index):
        h_list = []
        h = x

        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:  
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(h, edge_index)
        return h_list[-1], h 