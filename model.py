import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm

class NodeEmbeddingNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # per i residual
        self.dropout = nn.Dropout(p=dropout)

        # Primo layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))  # proiezione residual

        # Layer intermedi
        for i in range(1, num_layers):
            if i % 2 == 1:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
            self.res_projs.append(nn.Identity())  # residual layer dimensioni uguali

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

            # Residual con proiezione se serve
            if x_res.size(1) != x.size(1):
                x_res = self.res_projs[i](x_res)
            x = x + x_res
            x = self.dropout(x)

        # Normalizza embeddings finale
        x = F.normalize(x, p=2, dim=-1)
        return x


def unsupervised_loss(z, pos_edge_index, neg_edge_index):
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss