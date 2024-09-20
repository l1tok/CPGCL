import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            support = self.act(F.linear(features, self.weight)) 
        else:
            support = F.linear(features, self.weight) 
        output = torch.mm(adj, support)
        return output
    
class GCN(nn.Module):

    def __init__(self, n_input, hidden_dim1, hidden_dim2, z_dim, n_clusters):
        super(GCN, self).__init__()
        self.gnn_1 = GNNLayer(n_input, hidden_dim1)
        self.gnn_2 = GNNLayer(hidden_dim1, hidden_dim2)
        self.gnn_3 = GNNLayer(hidden_dim2, z_dim)
        self.clu =  nn.Sequential(nn.Linear(z_dim, n_clusters),
                            nn.Softmax())

    def forward(self, x, adj):
        z_1 = self.gnn_1(x, adj, active=True)
        z_2 = self.gnn_2(z_1, adj, active=True)
        z_out = self.gnn_3(z_2, adj, active=False)
        c_out = self.clu(z_out)
        return z_out, c_out
    