import torch
from torch_sparse import SparseTensor
from torch import Tensor
import torch_sparse
from typing import List, Tuple

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


def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

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

    
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

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


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

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

def de_plus_finder(adj, edges, mask_target=False):
    if mask_target:
        undirected_edges = torch.cat((edges, edges.flip(0)), dim=-1)
        target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj.sizes())
        adj, _ = spmnotoverlap_(adj, target_adj)
    # find 1,2 hops of target nodes
    l_1_1, l_1_not1, l_not1_1 = adjoverlap(adj[edges[0]], adj[edges[1]], calresadj=True) # not 1 == (dist=0) U dist(>=2)
    adj2_walks = adj @ adj
    adj2_return, _  = spmnotoverlap_(adj2_walks, adj)

    _, l_2_not2, l_not2_2 = adjoverlap(adj2_return[edges[0]], adj2_return[edges[1]], calresadj=True) 
    # not 2 == (dist=1) U dist(>2)
    # not include dist=0 because adj2_return will return with dist=0

    adj2, _ = spmnotoverlap_(adj2_return, SparseTensor.eye(adj.size(0), adj.size(1)).to(adj.device()))
    l_2_2, _, _ = adjoverlap(adj2[edges[0]], adj2[edges[1]], calresadj=True)

    l_1_2, l_1_not2, l_not1_2 = adjoverlap(adj[edges[0]], adj2[edges[1]], calresadj=True) # not also includes dist=0
    l_2_1, l_2_not1, l_not2_1 = adjoverlap(adj2[edges[0]], adj[edges[1]], calresadj=True)

    l_1_0inf = adjoverlap(l_1_not1, l_1_not2)
    l_0inf_1 = adjoverlap(l_not1_1, l_not2_1)

    remove_0s = SparseTensor.from_edge_index(
        torch.stack([torch.arange(edges.size(0)).repeat_interleave(2), edges.t().reshape(-1)]),
        sparse_sizes=(edges.size(0), adj.size(1)))
    l_1_inf,_ = spmnotoverlap_(l_1_0inf, remove_0s)
    l_inf_1,_ = spmnotoverlap_(l_0inf_1, remove_0s)

    l_2_inf = adjoverlap(l_2_not2, l_2_not1)
    l_inf_2 = adjoverlap(l_not2_2, l_not1_2)

    return l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2


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
    l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges)
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
    l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges, True)
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
    l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2 = de_plus_finder(adj, edges)
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







