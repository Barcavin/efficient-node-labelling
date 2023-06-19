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
from torch_geometric.transforms import (BaseTransform, Compose,
                                        NormalizeFeatures, RandomLinkSplit,
                                        ToDevice, ToUndirected)
from torch_geometric.utils import (add_self_loops, degree,
                                   from_scipy_sparse_matrix, index_to_mask,
                                   is_undirected, negative_sampling,
                                   to_undirected, train_test_split_edges, coalesce)                         


def get_dataset(root, name: str):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root, transform=transform)
        return dataset

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
        dataset = dataset_class(root, name=name, transform=ToUndirected())
    else:
        dataset = load_unsplitted_data(root, name)
        dataset = [dataset]
    return dataset

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


def do_edge_split(data, val_ratio=0.05, test_ratio=0.1, fe_ratio=0):
    if val_ratio==0:
        # force to generate some valid edge
        val_ratio_use=0.05
    else:
        val_ratio_use=val_ratio
    split = RandomLinkSplit(num_val=val_ratio_use,
                            num_test=test_ratio,
                            is_undirected=True,
                            split_labels=True,
                            add_negative_train_samples=False)
    train,val,test = split(data)
    # train.edge_index only train true edge
    # val.edge_index only train true edge
    # test.edge_index train+val true edge

    if val_ratio==0:
        train.edge_index = test.edge_index.clone()
        val.edge_index = test.edge_index.clone()
        train.pos_edge_label_index = torch.cat([train.pos_edge_label_index, val.pos_edge_label_index.clone()],axis=1)
        train.pos_edge_label = torch.cat([train.pos_edge_label, val.pos_edge_label.clone()])

    # split_edge has shape num_edges x 2
    split_edge = {"valid":{"edge":val.pos_edge_label_index.t(),
                                "edge_neg":val.neg_edge_label_index.t()},
        "test":{"edge":test.pos_edge_label_index.t(),
                        "edge_neg":test.neg_edge_label_index.t()},
        "train":{}}
    # print(f"train: {train.pos_edge_label_index.shape[1]}")
    # print(f"valid: {val.pos_edge_label_index.shape[1]}")
    # print(f"test: {test.pos_edge_label_index.shape[1]}")
    split_edge["train"]["edge"] = train.pos_edge_label_index.t()
    data = train
    return data, split_edge


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
    if file_path.exists():
        data, split_edge = torch.load(file_path)
        print(f"load split edges from {file_path}")
    else:
        dataset = get_dataset(root, name)
        original = dataset[0]
        data, split_edge = do_edge_split(original, val_ratio=val_ratio, test_ratio=test_ratio)
        torch.save((data, split_edge), file_path)
        print(f"save split edges to {file_path}")
    print("-"*20)
    print(f"train: {split_edge['train']['edge'].shape[0]}")
    print(f"{split_edge['train']['edge'][:10,:]}")
    print(f"valid: {split_edge['valid']['edge'].shape[0]}")
    print(f"test: {split_edge['test']['edge'].shape[0]}")
    print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
    return data, split_edge


def make_edge_index(pos_train_edges):
    """
        pos_train_edges: torch.tensor, shape [num_edges, 2]
    """
    edge_index = torch.cat([pos_train_edges, pos_train_edges.flip(1)], dim=0).t()
    return edge_index


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
        input_size = data.x.shape[1]
    return data, input_size

def create_input(data):
    if hasattr(data, 'emb') and data.emb is not None:
        x = data.emb.weight
    else:
        x = data.x
    return x
    