import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric import datasets
from torch_geometric.data import Data
from torch_geometric.transforms import (BaseTransform, Compose, ToSparseTensor,
                                        NormalizeFeatures, RandomLinkSplit,
                                        ToDevice, ToUndirected)
from torch_geometric.utils import (add_self_loops, degree,
                                   from_scipy_sparse_matrix, index_to_mask,
                                   is_undirected, negative_sampling,
                                   to_undirected, train_test_split_edges, coalesce)
from torch_sparse import SparseTensor                         


def get_dataset(root, name: str, use_valedges_as_input=False, year=-1):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=dataset, root=root)
        data = dataset[0]
        """
            SparseTensor's value is NxNx1 for collab. due to edge_weight is |E|x1
            NeuralNeighborCompletion just set edge_weight=None
            ELPH use edge_weight
        """
        if 'edge_weight' in data:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

        split_edge = dataset.get_edge_split()
        if name == 'ogbl-collab' and year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, year)
        print("-"*20)
        print(f"train: {split_edge['train']['edge'].shape[0]}")
        print(f"{split_edge['train']['edge'][:10,:]}")
        print(f"valid: {split_edge['valid']['edge'].shape[0]}")
        print(f"test: {split_edge['test']['edge'].shape[0]}")
        print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
        data = ToSparseTensor(remove_edge_index=False)(data)
        # Use training + validation edges for inference on test set.
        if use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
                                                    sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
            data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t
        return data, split_edge

    pyg_dataset_dict = {
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'Citeseer'),
        'pubmed': (datasets.Planetoid, 'Pubmed'),
        'cs': (datasets.Coauthor, 'CS'),
        'physics': (datasets.Coauthor, 'physics'),
        'computers': (datasets.Amazon, 'Computers'),
        'photos': (datasets.Amazon, 'Photo')
    }

    # assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    if name in pyg_dataset_dict:
        dataset_class, name = pyg_dataset_dict[name]
        data = dataset_class(root, name=name, transform=ToUndirected())[0]
    else:
        data = load_unsplitted_data(root, name)
    return data, None

def load_unsplitted_data(root,name):
    # read .mat format files
    data_dir = root + '/{}.mat'.format(name)
    # print('Load data from: '+ data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index,num_nodes = torch.max(edge_index).item()+1)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    return data

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# random split dataset
def randomsplit(data, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei

    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_data_split(root, name: str, val_ratio, test_ratio, run=0):
    data_folder = Path(root) / name
    data_folder.mkdir(parents=True, exist_ok=True)
    file_path = data_folder / f"split{run}_{int(100*val_ratio)}_{int(100*test_ratio)}.pt"
    data,_ = get_dataset(root, name)
    if file_path.exists():
        split_edge = torch.load(file_path)
        print(f"load split edges from {file_path}")
    else:
        split_edge = randomsplit(data)
        torch.save(split_edge, file_path)
        print(f"save split edges to {file_path}")
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    data.num_features = data.x.shape[0] if data.x is not None else 0
    print("-"*20)
    print(f"train: {split_edge['train']['edge'].shape[0]}")
    print(f"{split_edge['train']['edge'][:10,:]}")
    print(f"valid: {split_edge['valid']['edge'].shape[0]}")
    print(f"test: {split_edge['test']['edge'].shape[0]}")
    print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
    return data, split_edge


def data_summary(name: str, data: Data, header=False, latex=False):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    avg_degree = num_edges / num_nodes
    max_degree = degree(data.edge_index[0], num_nodes, dtype=torch.long).max().item()
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    if data.x is not None:
        attr_dim = data.x.shape[1]
    else:
        attr_dim = '-' # no attribute

    if latex:
        latex_str = ""
        if header:
            latex_str += r"""
            \begin{table*}[ht]
            \begin{center}
            \resizebox{0.85\textwidth}{!}{
            \begin{tabular}{lcccccc}
                \toprule
                \textbf{Dataset} & \textbf{\#Nodes} & \textbf{\#Edges} & \textbf{Avg. node deg.} & \textbf{Max. node deg.} & \textbf{Density} & \textbf{Attr. Dimension}\\
                \midrule"""
        latex_str += f"""
                \\textbf{{{name}}}"""
        latex_str += f""" & {num_nodes} & {num_edges} & {avg_degree:.2f} & {max_degree} & {density*100:.4f}\% & {attr_dim} \\\\"""
        latex_str += r"""
                \midrule"""
        if header:
            latex_str += r"""
            \bottomrule
            \end{tabular}
            }
            \end{center}
            \end{table*}"""
        print(latex_str)
    else:
        print("-"*30+'Dataset and Features'+"-"*40)
        print("{:<10}|{:<10}|{:<10}|{:<15}|{:<15}|{:<10}|{:<15}"\
            .format('Dataset','#Nodes','#Edges','Avg. node deg.','Max. node deg.', 'Density','Attr. Dimension'))
        print("-"*90)
        print("{:<10}|{:<10}|{:<10}|{:<15.2f}|{:<15}|{:<9.4f}%|{:<15}"\
            .format(name, num_nodes, num_edges, avg_degree, max_degree, density*100, attr_dim))
        print("-"*90)

def initialize(data, method):
    if data.x is None:
        if method == 'one-hot':
            data.x = F.one_hot(torch.arange(data.num_nodes),num_classes=data.num_nodes).float()
            input_size = data.num_nodes
        elif method == 'trainable':
            node_emb_dim = 512
            emb = torch.nn.Embedding(data.num_nodes, node_emb_dim)
            data.emb = emb
            input_size = node_emb_dim
        else:
            raise NotImplementedError
    else:
        input_size = data.num_features
    return data, input_size

def initial_embedding(data, hidden_channels, device):
    embedding= torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
    torch.nn.init.xavier_uniform_(embedding.weight)
    return embedding


def create_input(data):
    if hasattr(data, 'emb') and data.emb is not None:
        x = data.emb.weight
    else:
        x = data.x
    return x


# adopted from "https://github.com/melifluos/subgraph-sketching/tree/main"
def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge
    