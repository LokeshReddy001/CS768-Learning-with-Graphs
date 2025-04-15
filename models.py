import torch
import torch.nn as nn
import torch.nn.functional as F

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