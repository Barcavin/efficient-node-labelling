import math
from typing import List, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch_sparse import SparseTensor, matmul

import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data

from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph, to_edge_index

import torchhd

class DotHash(torch.nn.Module):
    def __init__(self, dim: int=1024, torchhd_style=True, prop_type="prop_only",
                 minimum_degree_onehot: int=-1):
        super().__init__()
        self.dim = dim
        self.torchhd_style = torchhd_style
        self.prop_type = prop_type
        self.cached_two_hop_adj = None
        self.minimum_degree_onehot = minimum_degree_onehot

    def forward(self, edges: Tensor, adj_t: SparseTensor, node_weight: Tensor=None):
        if self.prop_type == "prop_only":
            return self.propagation_only(edges, adj_t, node_weight)
        elif self.prop_type == "exact":
            return self.propagation(edges, adj_t, node_weight)
        elif self.prop_type == "combine":
            exact = self.propagation(edges, adj_t, node_weight)
            prop_only = self.propagation_only(edges, adj_t, node_weight)
            return exact + prop_only

    def subgraph(self, edges: Tensor, adj_t: SparseTensor):
        row,col = edges
        nodes = torch.cat((row,col),dim=-1)
        edge_index,_ = to_edge_index(adj_t)
        subset, new_edge_index, inv, edge_mask = pyg_k_hop_subgraph(nodes, 2, edge_index=edge_index, 
                                                                    num_nodes=adj_t.size(0), relabel_nodes=True)
        # subset[inv] = nodes. The new node id is based on `subset`'s order.
        # inv means the new idx (in subset) of the old nodes in `nodes`
        new_adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], 
                                 sparse_sizes=(subset.size(0), subset.size(0)))
        new_edges = inv.view(2,-1)
        return new_adj_t, new_edges

    def get_random_node_vectors(self, adj_t: SparseTensor, node_weight) -> Tensor:
        num_nodes = adj_t.size(0)
        device = adj_t.device()
        if self.minimum_degree_onehot > 0:
            degree = adj_t.sum(dim=1)
            nodes_to_one_hot = degree >= self.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            warnings.warn(f"number of nodes to one-hot: {one_hot_dim}")
            embedding = torch.zeros(num_nodes, one_hot_dim + self.dim, device=device)
            one_hot_embedding = F.one_hot(torch.arange(0, one_hot_dim)).float().to(device)
            embedding[nodes_to_one_hot,:one_hot_dim] = one_hot_embedding
        else:
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            nodes_to_one_hot = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            one_hot_dim = 0

        if self.torchhd_style:
            scale = math.sqrt(1 / self.dim)
            node_vectors = torchhd.random(num_nodes - one_hot_dim, self.dim, device=device)
            node_vectors.mul_(scale)  # make them unit vectors
        else:
            node_vectors = F.normalize(torch.nn.init.normal_(torch.empty((num_nodes - one_hot_dim, self.dim), dtype=torch.float32, device=device)))
        embedding[~nodes_to_one_hot, one_hot_dim:] = node_vectors

        if node_weight is not None:
            node_weight = node_weight.unsqueeze(1) # Note: not sqrt here because it can cause problem for MLP when output is negative
                                                   # thus, it requires the MLP to approximate one more sqrt?
            embedding.mul_(node_weight)
        return embedding

    def get_two_hop_adj(self, adj_t):
        # adj_t = adj_t.fill_value_(1.0) # no need to fill value because of subgraph op
        one_and_two_hop_adj = adj_t @ adj_t
        adj_t_with_self_loop = adj_t.fill_diag(1)
        two_hop_adj = spmdiff_(one_and_two_hop_adj, adj_t_with_self_loop)
        return adj_t, two_hop_adj

    def propagation(self, edges: Tensor, adj_t: SparseTensor, node_weight=None):
        # get the 2-hop subgraph of the target edges
        adj_t, edges = self.subgraph(edges, adj_t)
        x = self.get_random_node_vectors(adj_t, node_weight=node_weight)

        one_hop_adj, two_hop_adj = self.get_two_hop_adj(adj_t)
        subset = edges.view(-1) # flatten the target nodes [row, col]

        # size: [(2 x num_target_edges(row,col)) , total_num_nodes_in_2_hop_subgraph ]
        one_hop_adj = one_hop_adj[subset]
        two_hop_adj = two_hop_adj[subset]

        ##### no cache for testing, because the adj_t may change when use_val_as_input is True #####
        # if self.training: # training always requires compute especially for target input mask
        #     one_hop_adj, two_hop_adj = self.get_two_hop_adj(adj_t)
        # else: # testing
        #     if self.cached_two_hop_adj is None:
        #         # caching
        #         one_hop_adj, two_hop_adj = self.get_two_hop_adj(adj_t)
        #         self.cached_two_hop_adj = two_hop_adj
        #     else:
        #         # load cache
        #         one_hop_adj = adj_t.fill_value_(1.0)
        #         two_hop_adj = self.cached_two_hop_adj
            

        degree_one_hop = one_hop_adj.sum(dim=1)
        degree_one_hop = degree_one_hop.view(2, edges.size(1))
        degree_two_hop = two_hop_adj.sum(dim=1)
        degree_two_hop = degree_two_hop.view(2, edges.size(1))

        one_hop_x = matmul(one_hop_adj, x) # size: [(2 x num_target_edges(row,col)) , dim]
        two_hop_x = matmul(two_hop_adj, x)

        one_hop_x = one_hop_x.view(2, edges.size(1), -1)
        two_hop_x = two_hop_x.view(2, edges.size(1), -1)

        count_1_1 = dot_product(one_hop_x[0,:,:], one_hop_x[1,:,:])
        count_1_2 = dot_product(one_hop_x[0,:,:] , two_hop_x[1,:,:]) + dot_product(two_hop_x[0,:,:] , one_hop_x[1,:,:])
        count_2_2 = dot_product(two_hop_x[0,:,:] , two_hop_x[1,:,:])

        count_1_inf = degree_one_hop[0,:] + degree_one_hop[1,:] - 2 * count_1_1 - count_1_2
        count_2_inf = degree_two_hop[0,:] + degree_two_hop[1,:] - 2 * count_2_2 - count_1_2

        # count_1_1 = dot_product(one_hop_x[edges[0]] , one_hop_x[edges[1]])
        # count_1_2 = dot_product(one_hop_x[edges[0]] , two_hop_x[edges[1]]) + dot_product(two_hop_x[edges[0]] , one_hop_x[edges[1]])
        # count_2_2 = dot_product(two_hop_x[edges[0]] , two_hop_x[edges[1]])

        # count_1_inf = degree_one_hop[edges[0]] + degree_one_hop[edges[1]] - 2 * count_1_1 - count_1_2
        # count_2_inf = degree_two_hop[edges[0]] + degree_two_hop[edges[1]] - 2 * count_2_2 - count_1_2
        return count_1_1, count_1_2, count_2_2, count_1_inf, count_2_inf

    def propagation_only(self, edges: Tensor, adj_t: SparseTensor, node_weight=None):
        x = self.get_random_node_vectors(adj_t, node_weight=node_weight)
        # remove values from adj_t
        adj_t = adj_t.set_value(None)
        one_hop_x = matmul(adj_t, x)
        two_hop_x = matmul(adj_t, one_hop_x)
        degree_one_hop = adj_t.sum(dim=1)

        count_1_1 = dot_product(one_hop_x[edges[0]] , one_hop_x[edges[1]])
        count_1_2 = dot_product(one_hop_x[edges[0]] , two_hop_x[edges[1]]) + dot_product(two_hop_x[edges[0]] , one_hop_x[edges[1]])
        count_2_2 = dot_product((two_hop_x[edges[0]]-degree_one_hop[edges[0]].view(-1,1)*x[edges[0]]) , (two_hop_x[edges[1]]-degree_one_hop[edges[1]].view(-1,1)*x[edges[1]]))


        count_self_1_2 = dot_product(one_hop_x[edges[0]] , two_hop_x[edges[0]]) + dot_product(one_hop_x[edges[1]] , two_hop_x[edges[1]])
        return count_1_1, count_1_2, count_2_2, count_self_1_2


def dotproduct_naive(tensor1, tensor2):
    return (tensor1 * tensor2).sum(dim=-1)

def dotproduct_bmm(tensor1, tensor2):
    return torch.bmm(tensor1.unsqueeze(1), tensor2.unsqueeze(2)).view(-1)

dot_product = dotproduct_naive

def sparsesample(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > 0
    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand]

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask]

    ret = SparseTensor(row=samplerow.reshape(-1, 1).expand(-1, deg).flatten(),
                       col=samplecol.flatten(),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce().fill_value_(1.0)
    #print(ret.storage.value())
    return ret


def sparsesample2(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(
        row=torch.cat((samplerow, nosamplerow)),
        col=torch.cat((samplecol, nosamplecol)),
        sparse_sizes=adj.sparse_sizes()).to_device(
            adj.device()).fill_value_(1.0).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def elem2spm(element: Tensor, sizes: List[int], val: Tensor=None) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    if val is None:
        sp_tensor =  SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
            element.device).fill_value_(1.0)
    else:
        sp_tensor =  SparseTensor(row=row, col=col, value=val, sparse_sizes=sizes).to_device(
            element.device)
    return sp_tensor


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    val = spm.storage.value()
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem, val


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]
    '''
    nnz1 = adj1.nnz()
    element = torch.cat((adj1.storage.row(), adj2.storage.row()), dim=-1)
    element.bitwise_left_shift_(32)
    element[:nnz1] += adj1.storage.col()
    element[nnz1:] += adj2.storage.col()
    
    element = torch.sort(element, dim=-1)[0]
    mask = (element[1:] == element[:-1])
    retelem = element[:-1][mask]
    '''

    return elem2spm(retelem, adj1.sizes())


def spmnotoverlap_(adj1: SparseTensor,
                   adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()

    
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retelem2 = element2[torch.logical_not(matchedmask)]
    return elem2spm(retelem1, adj1.sizes()), elem2spm(retelem2, adj2.sizes())

def spmdiff_(adj1: SparseTensor,
                   adj2: SparseTensor, keep_val=False) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()

    
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]
    
    if keep_val and val1 is not None:
        retval1 = val1[maskelem1]
        return elem2spm(retelem1, adj1.sizes(), retval1)
    else:
        return elem2spm(retelem1, adj1.sizes())



def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retoverlap = element1
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retoverlap = element2[matchedmask]
        retelem2 = element2[torch.logical_not(matchedmask)]
    sizes = adj1.sizes()
    return elem2spm(retoverlap,
                    sizes), elem2spm(retelem1,
                                     sizes), elem2spm(retelem2, sizes)


def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    """
        returned sparse matrix shaped as [tarei.size(0), num_nodes]
        where each row represent the corresponding target edge,
        and each column represent whether that target edge has such a neighbor.
    """
    # a wrapper for functions above.
    if calresadj:
        adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
        if ressampledeg > 0:
            adjres1 = sparsesample_reweight(adjres1, ressampledeg)
            adjres2 = sparsesample_reweight(adjres2, ressampledeg)
        return adjoverlap, adjres1, adjres2
    else:
        adjoverlap = spmoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    return adjoverlap

def de_plus_finder(adj, edges, cached_adj2_return=None, cached_adj2=None):
    # if mask_target:
    #     undirected_edges = torch.cat((edges, edges.flip(0)), dim=-1)
    #     target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj.sizes())
    #     adj = spmdiff_(adj, target_adj)
    # find 1,2 hops of target nodes
    l_1_1, l_1_not1, l_not1_1 = adjoverlap(adj[edges[0]], adj[edges[1]], calresadj=True) # not 1 == (dist=0) U dist(>=2)
    if cached_adj2_return is None:
        adj2_walks = adj @ adj
        adj2_return  = spmdiff_(adj2_walks, adj)
    else:
        adj2_return = cached_adj2_return

    l_2_not2, l_not2_2 = spmnotoverlap_(adj2_return[edges[0]], adj2_return[edges[1]]) 
    # not 2 == (dist=1) U dist(>2)
    # not include dist=0 because adj2_return will return with dist=0

    if cached_adj2 is None:
        adj2 = spmdiff_(adj2_return, SparseTensor.eye(adj.size(0), adj.size(1)).to(adj.device()))
    else:
        adj2 = cached_adj2
    l_2_2 = adjoverlap(adj2[edges[0]], adj2[edges[1]])

    l_1_2, l_1_not2, l_not1_2 = adjoverlap(adj[edges[0]], adj2[edges[1]], calresadj=True) # not also includes dist=0
    l_2_1, l_2_not1, l_not2_1 = adjoverlap(adj2[edges[0]], adj[edges[1]], calresadj=True)

    l_1_0inf = adjoverlap(l_1_not1, l_1_not2)
    l_0inf_1 = adjoverlap(l_not1_1, l_not2_1)

    l_0_0 = SparseTensor.from_edge_index(
        torch.stack([torch.arange(edges.size(1)).repeat_interleave(2).to(edges.device), edges.t().reshape(-1)]),
        sparse_sizes=(edges.size(1), adj.size(1)))
    l_1_inf = spmdiff_(l_1_0inf, l_0_0)
    l_inf_1 = spmdiff_(l_0inf_1, l_0_0)

    l_2_inf = adjoverlap(l_2_not2, l_2_not1)
    l_inf_2 = adjoverlap(l_not2_2, l_not1_2)

    return (l_0_0, l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2), (adj2_return, adj2)


def isSymmetric(mat):
    N = mat.shape[0]
    for i in range(N):
        for j in range(N):
            if (mat[i][j] != mat[j][i]):
                return False
    return True

def check_all(pred, real):
    pred = pred.to_dense().numpy()
    real = real.to_dense().numpy()
    assert (pred == real).all()


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res

def k_hop_subgraph(src, dst, num_hops, A):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    return nodes, subgraph, dists

def construct_pyg_graph(node_ids, adj, dists, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'drnl_plus':  # DRNL
        z = drnl_node_labeling_plus(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z>100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(x=None, edge_index=edge_index, edge_weight=edge_weight, z=z, 
                    node_id=node_ids, num_nodes=num_nodes)
    return data

def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)

def drnl_node_labeling_plus(adj, src, dst):
    MAX_Z = 1000
    # Double Radius Node Labeling (DRNL) plus.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    
    dist2both_fill = torch.nan_to_num(dist2src,posinf=0) + torch.nan_to_num(dist2dst,posinf=0)
    z[torch.isnan(z)] =  - dist2both_fill[torch.isnan(z)] # last z to denote those 0s to distance to one of the end nodes

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def seal_extractor(src, dst, num_hops, A:SparseTensor, node_label='drnl'):
    # SparseTensor to ssp.csr
    A = A.to_scipy(layout='csr')
    nodes, subgraph, dists = k_hop_subgraph(src, dst, num_hops, A)
    data = construct_pyg_graph(nodes, subgraph, dists, node_label)
    return data

if __name__ == "__main__":
    adj1 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]]))
    adj2 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 3, 1, 2, 3], [0, 1, 1, 2, 3]]))
    adj3 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 1,  2, 2, 2,2, 3, 3, 3], [1, 0,  2,3,4, 5, 4, 5, 6]]))
    print(spmnotoverlap_(adj1, adj2))
    print(spmoverlap_(adj1, adj2))
    print(spmoverlap_notoverlap_(adj1, adj2))
    print(sparsesample2(adj3, 2))
    print(sparsesample_reweight(adj3, 2))

    print('-'*100)
    print("test de_plus_finder")
    "https://www.researchgate.net/figure/a-A-graph-with-six-nodes-and-seven-edges-b-A-adjacency-matrix-D-degree-matrix_fig3_339763754"
    adj = SparseTensor.from_dense(
        torch.LongTensor(
            [[0,1,0,0,0,1],
             [1,0,1,0,0,1],
             [0,1,0,1,1,0],
             [0,0,1,0,1,0],
             [0,0,1,1,0,0],
             [1,1,0,0,0,0]]
            ))
    print(adj)
    edges = torch.LongTensor([[0,2],[1,3]])
    print(f"edges: {edges}")
    _,l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges)
    print(f"l_1_1: {l_1_1}")
    print(f"l_1_2: {l_1_2}")
    print(f"l_2_1: {l_2_1}")
    print(f"l_1_inf: {l_1_inf}")
    print(f"l_inf_1: {l_inf_1}")
    print(f"l_2_2: {l_2_2}")
    print(f"l_2_inf: {l_2_inf}")
    print(f"l_inf_2: {l_inf_2}")
    l_1_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,1],
             [5,4]]
        ), sparse_sizes=(2,6))
    l_1_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[1],
             [1]]
        ), sparse_sizes=(2,6))
    l_2_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0],
             [2]]
        ), sparse_sizes=(2,6))
    l_1_inf_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,6))
    l_inf_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,6))
    l_2_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,6))
    l_2_inf_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[1,1],
             [0,5]]
        ), sparse_sizes=(2,6))
    l_inf_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,0],
             [3,4]]
        ), sparse_sizes=(2,6))
    check_all(l_1_1, l_1_1_true)
    check_all(l_1_2, l_1_2_true)
    check_all(l_2_1, l_2_1_true)
    check_all(l_1_inf, l_1_inf_true)
    check_all(l_inf_1, l_inf_1_true)
    check_all(l_2_2, l_2_2_true)
    check_all(l_2_inf, l_2_inf_true)
    check_all(l_inf_2, l_inf_2_true)

    print('-'*100)
    print("remove target edges")
    _,l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges, True)
    print(f"l_1_1: {l_1_1}")
    print(f"l_1_2: {l_1_2}")
    print(f"l_2_1: {l_2_1}")
    print(f"l_1_inf: {l_1_inf}")
    print(f"l_inf_1: {l_inf_1}")
    print(f"l_2_2: {l_2_2}")
    print(f"l_2_inf: {l_2_inf}")
    print(f"l_inf_2: {l_inf_2}")
    l_1_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,1],
             [5,4]]
        ), sparse_sizes=(2,6))
    l_1_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,6))
    l_2_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,6))
    l_1_inf_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[1],
             [1]]
        ), sparse_sizes=(2,6))
    l_inf_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0],
             [2]]
        ), sparse_sizes=(2,6))
    l_2_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,6))
    l_2_inf_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[1],
             [5]]
        ), sparse_sizes=(2,6))
    l_inf_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0],
             [4]]
        ), sparse_sizes=(2,6))
    check_all(l_1_1, l_1_1_true)
    check_all(l_1_2, l_1_2_true)
    check_all(l_2_1, l_2_1_true)
    check_all(l_1_inf, l_1_inf_true)
    check_all(l_inf_1, l_inf_1_true)
    check_all(l_2_2, l_2_2_true)
    check_all(l_2_inf, l_2_inf_true)
    check_all(l_inf_2, l_inf_2_true)


    print('-'*100)
    "https://www.researchgate.net/figure/The-graph-shown-in-a-has-its-adjacency-matrix-in-b-A-connection-between-two-nodes-is_fig6_291821895"
    adj = SparseTensor.from_dense(
        torch.LongTensor(
            [[0,0,1,1,0,0,0,0,1,0],
             [0,0,0,0,0,1,0,0,0,1],
             [1,0,0,0,0,1,1,1,0,0],
             [1,0,0,0,1,0,1,0,1,0],
             [0,0,0,1,0,0,0,0,0,1],
             [0,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,0,0,0,1,1,0],
             [0,0,1,0,0,0,1,0,0,1],
             [1,0,0,1,0,0,1,0,0,0],
             [0,1,0,0,1,0,0,1,0,0]]
            ))
    print(adj)
    assert isSymmetric(adj.to_dense().numpy())
    edges = torch.LongTensor([[0,2],[1,3]])
    print(f"edges: {edges}")
    _,l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges)
    print(f"l_1_1: {l_1_1}")
    print(f"l_1_2: {l_1_2}")
    print(f"l_2_1: {l_2_1}")
    print(f"l_1_inf: {l_1_inf}")
    print(f"l_inf_1: {l_inf_1}")
    print(f"l_2_2: {l_2_2}")
    print(f"l_2_inf: {l_2_inf}")
    print(f"l_inf_2: {l_inf_2}")
    l_1_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[1,1],
             [0,6]]
        ), sparse_sizes=(2,10))
    l_1_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,1],
             [2,7]]
        ), sparse_sizes=(2,10))
    l_2_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,1],
             [5,8]]
        ), sparse_sizes=(2,10))
    l_1_inf_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,0,1],
             [3,8,5]]
        ), sparse_sizes=(2,10))
    l_inf_1_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,1],
             [9,4]]
        ), sparse_sizes=(2,10))
    l_2_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,0,1],
             [4,7,9]]
        ), sparse_sizes=(2,10))
    l_2_inf_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[0,1],
             [6,1]]
        ), sparse_sizes=(2,10))
    l_inf_2_true = SparseTensor.from_edge_index(
        torch.LongTensor(
            [[],
             []]
        ), sparse_sizes=(2,10))
    check_all(l_1_1, l_1_1_true)
    check_all(l_1_2, l_1_2_true)
    check_all(l_2_1, l_2_1_true)
    check_all(l_1_inf, l_1_inf_true)
    check_all(l_inf_1, l_inf_1_true)
    check_all(l_2_2, l_2_2_true)
    check_all(l_2_inf, l_2_inf_true)
    check_all(l_inf_2, l_inf_2_true)







