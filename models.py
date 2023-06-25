import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (APPNP, GATConv, GCNConv, SAGEConv,
                                global_add_pool, global_max_pool,
                                global_mean_pool, global_sort_pool)
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_add

from Conv import Sage_conv
from node_label import de_plus_finder, DotHash


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

    def forward(self, feats, adj_t=None):
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
                 dropout, use_feature=True, embedding=None):
        super(GCN, self).__init__()

        self.use_feature = use_feature
        self.embedding = embedding
        self.dropout = dropout
        self.input_size = 0
        if self.use_feature:
            self.input_size += in_channels
            self.feat_encode = get_encoder(in_channels, self.dropout)
        if self.embedding is not None:
            self.input_size += embedding.embedding_dim
            self.embedding_encode = get_encoder(embedding.embedding_dim, self.dropout)
        self.convs = torch.nn.ModuleList()
        
        if self.input_size > 0:
            self.convs.append(GCNConv(self.input_size, hidden_channels, cached=False))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, cached=False))
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.input_size > 0:
            xs = []
            if self.use_feature:
                xs.append(self.feat_encode(x))
            if self.embedding is not None:
                xs.append(self.embedding_encode(self.embedding.weight))
            x = torch.cat(xs, dim=1)
            for conv in self.convs:
                x = conv(x, adj_t)
                # x = F.relu(x) # FIXME: not using nonlinearity in Sketching
                x = F.dropout(x, p=self.dropout, training=self.training)
            # x = self.convs[-1](x, adj_t) # Note: since it's not the last layer we may still apply dropout
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
                 dropout, num_hops=2, dgcnn=False, use_degree='none'):
        super(EfficientNodeLabelling, self).__init__()

        self.dropout = dropout
        self.num_hops = num_hops
        self.dgcnn = dgcnn
        self.in_channels = in_channels
        self.use_degree = use_degree
        if self.use_degree == 'mlp':
            self.node_weight_encode = MLP(2, in_channels + 1, 32, 1, dropout, norm_type="batch")

        self.max_z = 4
        self.z_embedding = nn.Embedding(self.max_z, hidden_channels)
        if self.dgcnn: # TODO: if enable DGCNN, GNN encoding may require tanh as discussed in the paper
                       #       Check why dgcnn sometimes run OOM
            self.k = 45 # TODO: dynamic determine the number of nodes to be held for each target edge
            self.struct_dim = 5
            total_latent_dim =  self.struct_dim + in_channels
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
            self.struct_dim = 5
            dense_dim = self.struct_dim + in_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(dense_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, 1))

        self.struct_encode = get_encoder(self.struct_dim, self.dropout)
        self.cached_adj2_return = None
        self.cached_adj2 = None
        # self.dothash = DotHash(torchhd_style=True, prop_type="exact")


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
        if self.training:
            (l_0_0, l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2), _ = de_plus_finder(adj, edges)
        else: # during testing we can cache the adj2
            if self.cached_adj2_return is None:
                (l_0_0, l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2), (adj2_return, adj2) = de_plus_finder(adj, edges, cached_adj2_return=None, cached_adj2=None)
                self.cached_adj2_return = adj2_return
                self.cached_adj2 = adj2
            else:
                (l_0_0, l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2), (adj2_return, adj2) = de_plus_finder(adj, edges, cached_adj2_return=self.cached_adj2_return, cached_adj2=self.cached_adj2)
        
        # concatenate the structural embedding
        # z = torch.LongTensor([(0,0)]*l_0_0.nnz()+
        #                [(1,1)]*l_1_1.nnz()+
        #                [(1,2)]*l_1_2.nnz()+
        #                [(2,1)]*l_2_1.nnz()+
        #                [(1,3)]*l_1_inf.nnz()+
        #                [(3,1)]*l_inf_1.nnz()+
        #                [(2,2)]*l_2_2.nnz()+
        #                [(2,3)]*l_2_inf.nnz()+
        #                [(3,2)]*l_inf_2.nnz()).to(x.device)
        # z_emb = self.z_embedding(z).sum(dim=1)
        # batch, node_ids = torch.concat([get_node_ids(l_0_0),
        #                                 get_node_ids(l_1_1),
        #                                 get_node_ids(l_1_2),
        #                                 get_node_ids(l_2_1),
        #                                 get_node_ids(l_1_inf),
        #                                 get_node_ids(l_inf_1),
        #                                 get_node_ids(l_2_2),
        #                                 get_node_ids(l_2_inf),
        #                                 get_node_ids(l_inf_2)], dim=1)
        # x_all =  z_emb
        if self.use_degree == 'none':
            node_weight = None
        elif self.use_degree == 'mlp': # 'mlp' for now
            xs = []
            if x is not None:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1,1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(-1)
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1,1).to(adj.device()).squeeze(-1)
            if self.use_degree == 'AA':
                node_weight = torch.reciprocal(torch.log(degree))
            elif self.use_degree == 'RA':
                node_weight = torch.reciprocal(degree)
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0, neginf=0.0)

        dim_size = edges.size(1)
        c_1_1 = get_count(l_1_1, dim_size, node_weight)
        c_1_2 = get_count(l_1_2, dim_size, node_weight) + get_count(l_2_1, dim_size, node_weight)
        c_1_inf = get_count(l_1_inf, dim_size, node_weight) + get_count(l_inf_1, dim_size, node_weight)
        c_2_2 = get_count(l_2_2, dim_size, node_weight)
        c_2_inf = get_count(l_2_inf, dim_size, node_weight) + get_count(l_inf_2, dim_size, node_weight)

        # count_1_1, count_1_2, count_2_2, count_1_inf, count_2_inf = self.dothash(edges, adj, node_weight=node_weight)

        out = torch.stack([c_1_1, c_1_2, c_1_inf, c_2_2, c_2_inf], dim=1).float()
        out = self.struct_encode(out)
        if self.in_channels > 0:
            x = torch.cat([x[edges[0]]*x[edges[1]], out], dim=1)
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


class DotProductLabelling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, num_hops=2, prop_type='exact', torchhd_style=True, use_degree='none'):
        super(DotProductLabelling, self).__init__()

        self.in_channels = in_channels
        self.dropout = dropout
        self.num_hops = num_hops
        self.prop_type = prop_type # "DP+exactly","DP+prop_only","DP+combine"
        self.torchhd_style=torchhd_style
        self.use_degree = use_degree
        if self.use_degree == 'mlp':
            self.node_weight_encode = MLP(2, in_channels + 1, 32, 1, dropout, norm_type="batch")
        if self.prop_type == 'prop_only':
            struct_dim = 4
        elif self.prop_type == 'exact':
            struct_dim = 5
        elif self.prop_type == 'combine':
            struct_dim = 8
        self.dothash = DotHash(torchhd_style=self.torchhd_style, prop_type=self.prop_type)
        self.struct_encode = get_encoder(struct_dim, self.dropout)

        dense_dim = struct_dim + in_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(dense_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, 1))


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
    
    def forward(self, x, adj, edges):
        """
        Args:
            x: [N, in_channels] node embedding after GNN
            adj: [N, N] adjacency matrix
            edges: [2, E] target edges
        """
        if self.use_degree == 'none':
            node_weight = None
        elif self.use_degree == 'mlp': # 'mlp' for now
            xs = []
            if x is not None:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1,1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(-1)
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1,1).to(adj.device()).squeeze(-1)
            if self.use_degree == 'AA':
                node_weight = torch.sqrt(torch.reciprocal(torch.log(degree)))
            elif self.use_degree == 'RA':
                node_weight = torch.sqrt(torch.reciprocal(degree))
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0, neginf=0.0)

        propped = self.dothash(edges, adj, node_weight=node_weight)
        out = torch.stack([*propped], dim=1)
        out = self.struct_encode(out)

        if self.in_channels > 0:
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






def get_node_ids(l:SparseTensor):
    row, col, _ = l.coo()
    return torch.stack([row, col], dim=0)

def get_count(l:SparseTensor, dim_size:int, weight=None):
    row,col = get_node_ids(l)
    if weight is None:
        weight = torch.ones_like(row)
    else:
        weight = weight[row]
    count = scatter_add(weight, row, dim_size=dim_size)
    return count

def get_encoder(dim, dropout):
    struct_encode = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
    return struct_encode