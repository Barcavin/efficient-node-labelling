import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (APPNP, GATConv, GCNConv, SAGEConv,
                                global_add_pool, global_max_pool,
                                global_mean_pool, global_sort_pool)
from torch_sparse import SparseTensor

from Conv import Sage_conv
from node_label import de_plus_finder


class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
        
class SAGE(torch.nn.Module):
    def __init__(self, data_name, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, norm_type="none"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.norm_type == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))            

        if data_name == "coauthor-physics":
            self.convs.append(Sage_conv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(Sage_conv(hidden_channels, hidden_channels))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_channels))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_channels))
            self.convs.append(Sage_conv(hidden_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_channels))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.norm_type != "none":
                    x = self.norms[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class APPNP_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, norm_type="none", alpha=0.1, k=10):
        super(APPNP_model, self).__init__()

        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNP(k, alpha, 0.)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        h = self.propagate(h, adj_t)
        return h

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout, norm_type="none"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=self.dropout))
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=self.dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj, edges):
        x_i = x[edges[0]]
        x_j = x[edges[1]]
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = torch.relu(x)
                hidden = x
                x = F.dropout(x, p=self.dropout, training=self.training)
            out = self.lins[-1](x)
        elif self.predictor == 'inner':
            hidden = x
            out = torch.sum(x, dim=-1)
        return out

class Teacher_LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(Teacher_LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)
        return torch.sigmoid(x)




class EfficientNodeLabelling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, num_hops=2, mask_target=False, dgcnn=False, use_feature=False):
        super(EfficientNodeLabelling, self).__init__()

        self.dropout = dropout
        self.num_hops = num_hops
        self.mask_target = mask_target
        self.dgcnn = dgcnn
        self.use_feature = use_feature
        if not self.use_feature:
            in_channels = 0

        self.max_z = 4
        self.z_embedding = nn.Embedding(self.max_z, hidden_channels)
        if self.dgcnn: # TODO: if enable DGCNN, GNN encoding may require tanh as discussed in the paper
                       #       Check why dgcnn sometimes run OOM
            self.k = 45 # TODO: dynamic determine the number of nodes to be held for each target edge
            total_latent_dim =  hidden_channels + in_channels
            conv1d_channels = [16, 32]
            conv1d_kws = [total_latent_dim, 5]
            self.conv1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                                conv1d_kws[0])
            self.maxpool1d = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1],
                                conv1d_kws[1], 1)
            dense_dim = int((self.k - 2) / 2 + 1)
            dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        else:
            dense_dim = hidden_channels + in_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(dense_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, 1))


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.z_embedding.reset_parameters()
        if self.dgcnn:
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()

    def forward(self, x, adj, edges):
        """
        Args:
            x: [N, in_channels] node embedding after GNN
            adj: [N, N] adjacency matrix
            edges: [2, E] target edges
        """
        
        l_0_0, l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges, mask_target=self.mask_target)
        # concatenate the structural embedding
        z = torch.LongTensor([(0,0)]*l_0_0.nnz()+
                       [(1,1)]*l_1_1.nnz()+
                       [(1,2)]*l_1_2.nnz()+
                       [(2,1)]*l_2_1.nnz()+
                       [(1,3)]*l_1_inf.nnz()+
                       [(3,1)]*l_inf_1.nnz()+
                       [(2,2)]*l_2_2.nnz()+
                       [(2,3)]*l_2_inf.nnz()+
                       [(3,2)]*l_inf_2.nnz()).to(x.device)
        z_emb = self.z_embedding(z).sum(dim=1)
        batch, node_ids = torch.concat([get_node_ids(l_0_0),
                                        get_node_ids(l_1_1),
                                        get_node_ids(l_1_2),
                                        get_node_ids(l_2_1),
                                        get_node_ids(l_1_inf),
                                        get_node_ids(l_inf_1),
                                        get_node_ids(l_2_2),
                                        get_node_ids(l_2_inf),
                                        get_node_ids(l_inf_2)], dim=1)
        x_all =  z_emb

        # x = x[node_ids]
        # x = torch.cat([x, z_emb], dim=1)
        
        if self.dgcnn:
            out = self.dgcnn_pooling(x_all, batch)
        else:
            out = global_mean_pool(x_all, batch)
        if self.use_feature:
            x_i = x[edges[0]]
            x_j = x[edges[1]]
            x = torch.cat([x_i*x_j, out], dim=1)
        else:
            x = out
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            hidden = x
            x = F.dropout(x, p=self.dropout, training=self.training)
        logit = self.lins[-1](x)
        return logit

    def dgcnn_pooling(self, all_x, batch):
        # Global pooling.
        x = global_sort_pool(all_x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        return x



def get_node_ids(l:SparseTensor):
    row, col, _ = l.coo()
    return torch.stack([row, col], dim=0)