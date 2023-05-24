import torch
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.models import GIN
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

class MLP(torch.nn.Module):
  def __init__(self, input_channels, hidden_channels, output_channels):
    super(MLP, self).__init__()

    self.input_channels = input_channels
    self.hidden_channels = hidden_channels
    self.output_channels = output_channels

    #transformation on the graph
    self.lin1 = Linear(input_channels, hidden_channels)
    self.lin2 = Linear(hidden_channels, hidden_channels)
    self.lin3 = Linear(hidden_channels, hidden_channels)

    self.linnews = Linear(input_channels, hidden_channels)

    self.lin0 = Linear(hidden_channels, hidden_channels)
    self.lin = Linear(2*hidden_channels, output_channels) #classifier

    self.reset_parameters()
    
  def reset_parameters(self):
    self.lin1.reset_parameters()
    self.lin2.reset_parameters()
    self.lin3.reset_parameters()
    self.linnews.reset_parameters()
    self.lin0.reset_parameters()
    self.lin.reset_parameters()
    

  def forward(self, x, batch):
    h = self.lin1(x).relu()
    #h = F.dropout(h, p=args.dropout_ratio, training=self.training)
    h = self.lin2(h).relu()
    #h = F.dropout(h, p=args.dropout_ratio, training=self.training)
    h = self.lin3(h).relu()

    h = global_max_pool(h, batch)

    h = self.lin0(h).relu()

    root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
    root = torch.cat([root.new_zeros(1), root + 1], dim=0)
    news = x[root]
    news = self.linnews(news).relu()
    out = self.lin(torch.cat([h, news], dim=-1))

    return torch.sigmoid(out)


class GCN(torch.nn.Module):
  def __init__(self, input_channels, hidden_channels, output_channels, num_gnn_layers):
    super(GCN, self).__init__()

    self.input_channels = input_channels
    self.hidden_channels = hidden_channels
    self.output_channels = output_channels
    self.num_layers = num_gnn_layers

    self.convs = torch.nn.ModuleList()

    for layer in range(num_gnn_layers):
      self.convs.append(GCNConv(input_channels, hidden_channels))

    self.linnews = Linear(input_channels, hidden_channels)

    self.lin2 = Linear(hidden_channels, hidden_channels) #after max pooling
    self.lin3 = Linear(2*hidden_channels, output_channels)

    self.reset_parameters()

  def reset_parameters(self):
    for conv in self.convs:
        conv.reset_parameters()
    self.linnews.reset_parameters()
    self.lin2.reset_parameters()
    self.lin3.reset_parameters()

  def forward(self, x, adj, batch):
    for idx in range(len(self.convs)):
      h = self.convs[idx](x, adj).relu()

    h = global_max_pool(h, batch)
    h = self.lin2(h).relu()

    root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
    root = torch.cat([root.new_zeros(1), root + 1], dim=0)
    news = x[root]
    news = self.linnews(news).relu()

    out = self.lin3(torch.cat([h, news], dim=-1))

    return torch.sigmoid(out)


class GraphSage(torch.nn.Module):
  def __init__(self, input_channels, hidden_channels, output_channels):
    super(GraphSage, self).__init__()
    
    self.input_channels = input_channels
    self.hidden_channels = hidden_channels
    self.output_channels = output_channels

    self.sage1 = SAGEConv(input_channels, hidden_channels, normalize=True)
    self.sage2 = SAGEConv(hidden_channels, hidden_channels, normalize = True)
    self.sage3 = SAGEConv(hidden_channels, hidden_channels, normalize = True)

    self.linnews = Linear(input_channels, hidden_channels)

    self.lin2 = Linear(hidden_channels, hidden_channels) #after max pooling
    self.lin3 = Linear(2*hidden_channels, output_channels)

    self.reset_parameters()
    
  def reset_parameters(self):
    self.sage1.reset_parameters()
    self.sage2.reset_parameters()
    self.sage3.reset_parameters()
    self.linnews.reset_parameters()
    self.lin2.reset_parameters()
    self.lin3.reset_parameters()

  def forward(self, x, adj, batch):
    h = self.sage1(x, adj).relu()
    h = self.sage2(h, adj).relu()
    h = self.sage3(h, adj).relu()

    h = global_max_pool(h, batch)
    h = self.lin2(h).relu()

    root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
    root = torch.cat([root.new_zeros(1), root + 1], dim=0)
    news = x[root]
    news = self.linnews(news).relu()

    out = self.lin3(torch.cat([h, news], dim=-1))

    return torch.sigmoid(out)


class GAT(torch.nn.Module):
  def __init__(self, input_channels, hidden_channels, output_channels):
    super(GAT, self).__init__()
    
    self.input_channels = input_channels
    self.hidden_channels = hidden_channels
    self.output_channels = output_channels

    self.Gatconv1 = GATConv(input_channels, hidden_channels)
    self.Gatconv2 = GATConv(hidden_channels, hidden_channels)
    self.Gatconv3 = GATConv(hidden_channels, hidden_channels)

    self.linnews = Linear(input_channels, hidden_channels)
    self.lin0 = Linear(hidden_channels, hidden_channels)
    self.lin1 = Linear(2*hidden_channels, output_channels)

    self.reset_parameters()

  def reset_parameters(self):
    self.Gatconv1.reset_parameters()
    self.Gatconv2.reset_parameters()
    self.Gatconv3.reset_parameters()
    self.linnews.reset_parameters()
    self.lin0.reset_parameters()
    self.lin1.reset_parameters()

  def forward(self, x, adj, batch):
    h = self.Gatconv1(x, adj).relu()
    h = self.Gatconv2(h, adj).relu()
    h = self.Gatconv3(h, adj).relu()

    h = global_max_pool(h, batch)
    h = self.lin0(h).relu()

    root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
    root = torch.cat([root.new_zeros(1), root + 1], dim=0)
    news = x[root]
    news = self.linnews(news).relu()

    out = self.lin1(torch.cat([h, news], dim=-1))

    return torch.sigmoid(out)


class Graph_Isomorphism_Network(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers):
      super(Graph_Isomorphism_Network, self).__init__()

      self.input_channels = input_channels
      self.hidden_channels = hidden_channels
      self.output_channels = output_channels
      self.num_layers = num_layers

      self.gnn1 = GIN(input_channels, hidden_channels, num_layers)

      self.linnews = Linear(input_channels, hidden_channels)

      self.lin2 = Linear(hidden_channels, hidden_channels) #after max pooling
      self.lin3 = Linear(2*hidden_channels, output_channels)

      self.reset_parameters()
      
    def reset_parameters(self):
      self.gnn1.reset_parameters()
      self.linnews.reset_parameters()
      self.lin2.reset_parameters()
      self.lin3.reset_parameters()

    def forward(self, x, adj, batch):
      h = self.gnn1(x, adj).relu()

      h = global_max_pool(h, batch)
      h = self.lin2(h).relu()

      root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
      root = torch.cat([root.new_zeros(1), root + 1], dim=0)
      news = x[root]
      news = self.linnews(news).relu()

      out = self.lin3(torch.cat([h, news], dim=-1))

      return torch.sigmoid(out)


