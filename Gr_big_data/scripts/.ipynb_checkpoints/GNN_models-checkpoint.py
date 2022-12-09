import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
#import torch_geometric.nn as geom_nn

class GCN_easy(torch.nn.Module):
    def __init__(self, dataset, activation = torch.nn.PReLU,
                 hidden_size = 16, out_size = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.activation = activation()
        self.conv2 = GCNConv(hidden_size, out_size)

    def forward(self, data):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node (for aggregation)
        """
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch_idx) # Average pooling
        #x = self.head(x)
        return x
    

class GCN_model(torch.nn.Module):
    def __init__(self, dataset, activation = torch.nn.PReLU,
                 n_hidden_layers = 2,
                 hidden_size = 16, out_size = 2):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.out_size = out_size
        
        self.conv_start = GCNConv(dataset.num_node_features, hidden_size)
        self.conv_end = GCNConv(hidden_size, out_size)
        
        self.hidden_layers =  torch.nn.ModuleList([GCNConv(hidden_size, hidden_size) for i in range(self.n_hidden_layers)])
        self.activation_layers =  torch.nn.ModuleList([activation() for i in range(self.n_hidden_layers + 1)]) # + start 
        

    def forward(self, data):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch

        x = self.conv_start(x, edge_index)
        x = self.activation_layers[0](x)
        #x = F.dropout(x, training=self.training)
        for i in range(self.n_hidden_layers):
            x = self.hidden_layers[i](x, edge_index)
            x = self.activation_layers[i+1](x)           
        x = self.conv_end(x, edge_index)
        x = global_mean_pool(x, batch_idx) # Average pooling
        #x = self.head(x)
        return x
        
    
    
'''    
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = torch.nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
''' 