#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
from utils import get_dataset
from models import get_count
from hashing import ElphHashes
from node_label import DotHash, de_plus_finder, spmdiff_
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import negative_sampling

from torch_sparse import SparseTensor

import argparse
from argparse import Namespace

from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

from utils import ( get_dataset, data_summary,
                   set_random_seeds, str2bool, get_data_split, initial_embedding)

# In[2]:


from functools import wraps
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int)
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--minimum_degree_onehot', type=int, default=100)

args = parser.parse_args()

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result, total_time
    return timeit_wrapper


# In[3]:


dataset = "ogbl-collab"
# batch_size = 512
# batch_size = 1024
batch_size = args.batch_size
sample=  50000
device = torch.device('cuda')
set_random_seeds(234)


# In[4]:


data = get_dataset("../efficient-node-labelling/data", dataset, year=2010)[0]
# data = get_dataset("../efficient-node-labelling/data", dataset)[0]
data = ToSparseTensor(remove_edge_index=False)(data)


# In[5]:


def mark_target(adj_t,edge):
    undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
    target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
    adj_t = spmdiff_(adj_t, target_adj, keep_val=False)
    return adj_t


# In[6]:


def get_edges(data, sample=None):
    row,col = data.edge_index
    pos_edge = data.edge_index[:,row<col]
    if sample is None:
        sample = pos_edge.shape[1]
    else:
        pos_edge = pos_edge[:,torch.randperm(pos_edge.shape[1])][:,:sample]
    neg_edge = negative_sampling(data.edge_index,num_neg_samples=sample)
    all_edge = torch.concat([pos_edge,neg_edge],axis=1)
    all_edge = all_edge[:,torch.randperm(all_edge.shape[1])]
    return all_edge


# In[7]:


edges = get_edges(data, sample)

print(f"scoring edges: {edges}")

# In[8]:


@timeit
def run_dothash(dothash_dim, data, edges, num_trials=1, minimum_degree_onehot=-1, torchhd=True):
    dothash = DotHash(dothash_dim, torchhd, prop_type='exact',minimum_degree_onehot=minimum_degree_onehot)
    dothash.train()
    c_1_1, c_1_2, c_2_2, c_1_inf, c_2_inf = [], [], [], [], []
    for perm in DataLoader(range(edges.size(1)), batch_size):
        edge = edges[:,perm]
        adj_t = mark_target(data.adj_t, edge)
        d = defaultdict(list)
        d_result = {}
        # count_1_1 = []
        # count_1_2 = []
        # count_2_2 = []
        # count_1_inf = []
        # count_2_inf = []
        for trial in range(num_trials):
            propped = dothash(edge,adj_t)
            for i,each in enumerate(propped):
                d[i].append(each)
            # count_1_1.append(count_1_1_tmp)
            # count_1_2.append(count_1_2_tmp)
            # count_2_2.append(count_2_2_tmp)
            # count_1_inf.append(count_1_inf_tmp)
            # count_2_inf.append(count_2_inf_tmp)
        for i,di in d.items():
            stack = torch.stack(di)
            mean = stack.mean(axis=0)
            d_result[i] = mean
        c_1_1.append(d_result[0])
        c_1_2.append(d_result[1])
        c_2_2.append(d_result[2])
        c_1_inf.append(d_result[3])
        c_2_inf.append(d_result[4])
    c_1_1 = torch.concat(c_1_1)
    c_1_2 = torch.concat(c_1_2)
    c_2_2 = torch.concat(c_2_2)
    c_1_inf = torch.concat(c_1_inf)
    c_2_inf = torch.concat(c_2_inf)
    return c_1_1, c_1_2, c_2_2, c_1_inf, c_2_inf


# In[9]:


data= data.to(device)
edges = edges.to(device)
# run_dothash(1024,data,edges,num_trials=1)


# In[10]:


@timeit
def run_count(data, edges):
    c_1_1, c_1_2, c_2_2, c_1_inf, c_2_inf = [], [], [], [], []
    node_weight=None
    for perm in DataLoader(range(edges.size(1)), batch_size):
        edge = edges[:,perm]
        adj_t = mark_target(data.adj_t, edge)
        (l_0_0, l_1_1, l_1_2, l_2_1, l_1_inf, l_inf_1, l_2_2, l_2_inf, l_inf_2), _ = de_plus_finder(adj_t, edge)
        dim_size = edge.size(1)
        c_1_1.append(get_count(l_1_1, dim_size, node_weight))
        c_1_2.append(get_count(l_1_2, dim_size, node_weight) + get_count(l_2_1, dim_size, node_weight))
        c_1_inf.append(get_count(l_1_inf, dim_size, node_weight) + get_count(l_inf_1, dim_size, node_weight))
        c_2_2.append(get_count(l_2_2, dim_size, node_weight))
        c_2_inf.append(get_count(l_2_inf, dim_size, node_weight) + get_count(l_inf_2, dim_size, node_weight))
    c_1_1 = torch.concat(c_1_1)
    c_1_2 = torch.concat(c_1_2)
    c_2_2 = torch.concat(c_2_2)
    c_1_inf = torch.concat(c_1_inf)
    c_2_inf = torch.concat(c_2_inf)
    return c_1_1, c_1_2, c_2_2, c_1_inf, c_2_inf


# In[11]:


# run_count(data,edges)


# In[ ]:





# In[12]:
from tqdm import tqdm

@timeit
def run_elph(minhash_num_perm, data, edges, init_once=False):
    args = Namespace(max_hash_hops = 2, 
                     floor_sf = 0,
                     minhash_num_perm = minhash_num_perm,
                     hll_p = 8,
                     use_zero_one = True)
    elph = ElphHashes(args)
    c_1_1, c_1_2, c_2_2, c_1_inf, c_2_inf = [], [], [], [], []
    if init_once:
        row,col,_ = data.adj_t.coo()
        edge_index = torch.stack([row,col])
        hashes, cards = elph.build_hash_tables(data.num_nodes, edge_index)
    for perm in DataLoader(range(edges.size(1)), batch_size):
        edge = edges[:,perm]
        if not init_once:
            adj_t = mark_target(data.adj_t, edge)
            row,col,_ = adj_t.coo()
            edge_index = torch.stack([row,col])
            hashes, cards = elph.build_hash_tables(data.num_nodes, edge_index)
        subgraph_features = elph.get_subgraph_features(edge.t(), hashes, cards)
        c_1_1.append(subgraph_features[:,0])
        c_1_2.append(torch.sum(subgraph_features[:, 1:3], dim=1))
        c_2_2.append(subgraph_features[:,3])
        c_1_inf.append(torch.sum(subgraph_features[:, 4:6], dim=1))
        c_2_inf.append(torch.sum(subgraph_features[:, 6:8], dim=1))
    c_1_1 = torch.concat(c_1_1)
    c_1_2 = torch.concat(c_1_2)
    c_2_2 = torch.concat(c_2_2)
    c_1_inf = torch.concat(c_1_inf)
    c_2_inf = torch.concat(c_2_inf)
    return c_1_1, c_1_2, c_2_2, c_1_inf, c_2_inf


# In[13]:


# data= data.to(device)
# edges = edges.to(device)
# run_elph(512,data,edges)


# In[14]:


def MSE(y,y_hat):
    return ((y-y_hat)**2).mean()


# In[15]:


def get_MSE(pred, true):
    labels = ["(1,1)","(1,2)","(2,2)","(1,inf)","(2,inf)"]
    result = {}
    for est, count, pos in zip(pred,true,labels):
        result[pos] = MSE(est,count).cpu().item()
    return result


# In[16]:


true_count,running_time = run_count(data,edges)
dim = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
model_name = ["DotHash","MPLP","ELPH"]
# minimum_degree_onehot = 50
minimum_degree_onehot = args.minimum_degree_onehot
# dim = [512,1024,2048,4096]
result = []
runs=[]
# for dim1 in dim:
#     for model_name1 in model_name:
#         runs.append((dim1, model_name1))

if args.model_name =="DotHash":
    dothash_est,running_time = run_dothash(args.dim,data,edges,minimum_degree_onehot=-1)
    result11 = get_MSE(dothash_est, true_count)
    result11["model"] = "DotHash"
    result11["dim"] = args.dim
    result11["time"] = running_time
    result.append(result11)
elif args.model_name == "MPLP":
    dothash_est,running_time = run_dothash(args.dim,data,edges,minimum_degree_onehot=minimum_degree_onehot)
    result11 = get_MSE(dothash_est, true_count)
    result11["model"] = "MPLP"
    result11["dim"] = args.dim
    result11["time"] = running_time
    result.append(result11)
#     # dothash_est,running_time = run_dothash(dim1,data,edges,minimum_degree_onehot=-1,torchhd=False)
#     # result11 = get_MSE(dothash_est, true_count)
#     # result11["model"] = "DotHash+normal"
#     # result11["dim"] = dim1
#     # result11["time"] = running_time
#     # result.append(result11)
    
#     # dothash_est,running_time = run_dothash(dim1,data,edges,minimum_degree_onehot=minimum_degree_onehot,torchhd=False)
#     # result11 = get_MSE(dothash_est, true_count)
#     # result11["model"] = "DotHash+normal++"
#     # result11["dim"] = dim1
#     # result11["time"] = running_time
#     # result.append(result11)

elif args.model_name == "ELPH":
    elph_est, running_time = run_elph(args.dim,data,edges,init_once=True)
    result11 = get_MSE(elph_est, true_count)
    result11["model"] = "ELPH"
    result11["dim"] = args.dim
    result11["time"] = running_time
    result.append(result11)


# # In[19]:
# check whether file exist
filename = "hash_compare.csv"
from pathlib import Path
my_file = Path(filename)
if my_file.exists():
    mode='a'
    header=False
else:
    mode='w'
    header=True
df = pd.DataFrame(result)
df.to_csv(my_file,index=False,header=header,mode=mode)