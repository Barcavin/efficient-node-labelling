import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (APPNP, GATConv, GCNConv, SAGEConv,
                                global_add_pool, global_max_pool,
                                global_mean_pool, global_sort_pool)
from torch_sparse import SparseTensor
# from torch_scatter import scatter_add
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add

from functools import partial

from node_label import de_plus_finder, NodeLabel
from typing import Iterable, Final


class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
        tailnormactdrop=False,
        affine=True,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.tailnormactdrop = tailnormactdrop
        self.affine = affine # the affine in batchnorm
        self.layers = []
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
            if tailnormactdrop:
                self.__build_normactdrop(self.layers, output_dim, dropout_ratio)
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            if tailnormactdrop:
                self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
        self.layers = nn.Sequential(*self.layers)

    def __build_normactdrop(self, layers, dim, dropout):
        if self.norm_type == "batch":
            layers.append(nn.BatchNorm1d(dim, affine=self.affine))
        elif self.norm_type == "layer":
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Dropout(dropout, inplace=True))
        layers.append(nn.ReLU(inplace=True))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, feats, adj_t=None):
        return self.layers(feats)

# Addpted from NCNC
class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, xdropout, use_feature=True, jk=False, gcn_name='gcn', embedding=None):
        super(GCN, self).__init__()

        self.use_feature = use_feature
        self.embedding = embedding
        self.dropout = dropout
        self.xdropout = xdropout
        self.input_size = 0
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
        if self.use_feature:
            self.input_size += in_channels
        if self.embedding is not None:
            self.input_size += embedding.embedding_dim
        self.convs = torch.nn.ModuleList()
        
        if self.input_size > 0:
            if gcn_name == 'gcn':
                conv_func = partial(GCNConv, cached=False)
            elif 'pure' in gcn_name:
                conv_func = partial(PureConv, aggr='gcn')
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) # nn.Identity()
            if ("pure" in gcn_name or num_layers==0):
                self.xemb.append(nn.Linear(self.input_size, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
                self.input_size = hidden_channels
            self.convs.append(conv_func(self.input_size, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv_func(hidden_channels, hidden_channels))
            self.convs.append(conv_func(hidden_channels, out_channels))


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, adj_t):
        if self.input_size > 0:
            xs = []
            if self.use_feature:
                xs.append(x)
            if self.embedding is not None:
                xs.append(self.embedding.weight)
            x = torch.cat(xs, dim=1)
            x = self.xemb(x)
            jkx = []
            for conv in self.convs:
                x = conv(x, adj_t)
                # x = F.relu(x) # FIXME: not using nonlinearity in Sketching
                if self.jk:
                    jkx.append(x)
            if self.jk: # JumpingKnowledge Connection
                jkx = torch.stack(jkx, dim=0)
                sftmax = self.jkparams.reshape(-1, 1, 1)
                x = torch.sum(jkx*sftmax, dim=0)
        return x
        
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, xdropout, use_feature=True, jk=False, embedding=None):
        super(SAGE, self).__init__()

        self.use_feature = use_feature
        self.embedding = embedding
        self.dropout = dropout
        self.xdropout = xdropout
        self.input_size = 0
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
        if self.use_feature:
            self.input_size += in_channels
        if self.embedding is not None:
            self.input_size += embedding.embedding_dim
        self.convs = torch.nn.ModuleList()
        
        if self.input_size > 0:
            conv_func = partial(SAGEConv, cached=False)
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) # nn.Identity()
            if num_layers==0:
                self.xemb.append(nn.Linear(self.input_size, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
                self.input_size = hidden_channels
            self.convs.append(conv_func(self.input_size, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv_func(hidden_channels, hidden_channels))
            self.convs.append(conv_func(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, adj_t):
        if self.input_size > 0:
            xs = []
            if self.use_feature:
                xs.append(x)
            if self.embedding is not None:
                xs.append(self.embedding.weight)
            x = torch.cat(xs, dim=1)
            x = self.xemb(x)
            jkx = []
            for conv in self.convs:
                x = conv(x, adj_t)
                # x = F.relu(x) # FIXME: not using nonlinearity in Sketching
                if self.jk:
                    jkx.append(x)
            if self.jk: # JumpingKnowledge Connection
                jkx = torch.stack(jkx, dim=0)
                sftmax = self.jkparams.reshape(-1, 1, 1)
                x = torch.sum(jkx*sftmax, dim=0)
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

    def forward(self, x, adj, edges, **kwargs):
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


class MPLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 feat_dropout, label_dropout, num_hops=2, prop_type='combine', signature_sampling='torchhd', use_degree='none',
                 signature_dim=1024, minimum_degree_onehot=-1, batchnorm_affine=True,
                 feature_combine="hadamard",adj2=False):
        super(MPLP, self).__init__()

        self.in_channels = in_channels
        self.feat_dropout = feat_dropout
        self.label_dropout = label_dropout
        self.num_hops = num_hops
        self.prop_type = prop_type # "MPLP+exactly","MPLP+prop_only","MPLP+combine"
        self.signature_sampling=signature_sampling
        self.use_degree = use_degree
        self.feature_combine = feature_combine
        self.adj2 = adj2
        if self.use_degree == 'mlp':
            self.node_weight_encode = MLP(2, in_channels + 1, 32, 1, feat_dropout, norm_type="batch", affine=batchnorm_affine)
        if self.prop_type in ['prop_only', 'precompute']:
            struct_dim = 8
        elif self.prop_type == 'exact':
            struct_dim = 5
        elif self.prop_type == 'combine':
            struct_dim = 15
        self.nodelabel = NodeLabel(signature_dim, signature_sampling=self.signature_sampling, prop_type=self.prop_type,
                               minimum_degree_onehot= minimum_degree_onehot)
        self.struct_encode = MLP(1, struct_dim, struct_dim, struct_dim, self.label_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)

        dense_dim = struct_dim + in_channels
        if in_channels > 0:
            if feature_combine == "hadamard":
                feat_encode_input_dim = in_channels
            elif feature_combine == "plus_minus":
                feat_encode_input_dim = in_channels * 2
            self.feat_encode = MLP(2, feat_encode_input_dim, in_channels, in_channels, self.feat_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)
        self.classifier = nn.Linear(dense_dim, 1)


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
    
    def forward(self, x, adj, edges, cache_mode=None, adj2=None):
        """
        Args:
            x: [N, in_channels] node embedding after GNN
            adj: [N, N] adjacency matrix
            edges: [2, E] target edges
            fast_inference: bool. If True, only caching the message-passing without calculating the structural features
        """
        if cache_mode is None and self.prop_type == "precompute":
            # when using precompute, forward always use cache_mode == 'use'
            cache_mode = 'use'
        if cache_mode in ["use","delete"]:
            # no need to compute node_weight
            node_weight = None
        elif self.use_degree == 'none':
            node_weight = None
        elif self.use_degree == 'mlp': # 'mlp' for now
            xs = []
            if self.in_channels > 0:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1,1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(-1) + 1 # like residual, can be learned as 0 if needed
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1,1).to(adj.device()).squeeze(-1) + 1 # degree at least 1. then log(degree) > 0.
            if self.use_degree == 'AA':
                node_weight = torch.sqrt(torch.reciprocal(torch.log(degree)))
            elif self.use_degree == 'RA':
                node_weight = torch.sqrt(torch.reciprocal(degree))
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0, neginf=0.0)

        if cache_mode in ["build","delete"]:
            propped = self.nodelabel(edges, adj, node_weight=node_weight, cache_mode=cache_mode)
            return
        else:
            propped = self.nodelabel(edges, adj, node_weight=node_weight, cache_mode=cache_mode, adj2=adj2)
        propped_stack = torch.stack([*propped], dim=1)
        out = self.struct_encode(propped_stack)

        if self.in_channels > 0:
            x_i = x[edges[0]]
            x_j = x[edges[1]]
            if self.feature_combine == "hadamard":
                x_ij = x_i * x_j
            elif self.feature_combine == "plus_minus":
                x_ij = torch.cat([x_i+x_j, torch.abs(x_i-x_j)], dim=1)
            x_ij = self.feat_encode(x_ij)
            x = torch.cat([x_ij, out], dim=1)
        else:
            x = out
        logit = self.classifier(x)
        return logit

    def precompute(self, adj):
        self(None, adj, None, cache_mode="build")
        return self





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



### Misc ###
class NaiveNodeLabelling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, num_hops=2, dgcnn=False, use_degree='none'):
        super(NaiveNodeLabelling, self).__init__()

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

        self.struct_encode = MLP(1, self.struct_dim, self.struct_dim, self.struct_dim, self.dropout, "batch", tailnormactdrop=True)
        self.cached_adj2_return = None
        self.cached_adj2 = None
        # self.nodelabel = NodeLabel(torchhd_style=True, prop_type="exact")


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

        # count_1_1, count_1_2, count_2_2, count_1_inf, count_2_inf = self.nodelabel(edges, adj, node_weight=node_weight)

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