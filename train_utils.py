import time
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score

from ogb.linkproppred import Evaluator
from torch_geometric.utils import negative_sampling

from tqdm import tqdm
from node_label import spmdiff_, QUERY_GRAPH_LABEL
from logger import Logger



def get_train_test(args):
    inference_datasets = args.inference_datasets.split(',')
    loggers = {}
    for one_dataset_name in inference_datasets + ['#']:
        loggers.update({
            f'Hits@10{one_dataset_name}': Logger(args.runs, args),
            f'Hits@20{one_dataset_name}': Logger(args.runs, args),
            f'Hits@50{one_dataset_name}': Logger(args.runs, args),
            f'Hits@100{one_dataset_name}': Logger(args.runs, args),
            f'AUC{one_dataset_name}': Logger(args.runs, args),
        })
    evaluator = Evaluator(name='ogbl-ddi')
    return train_hits, test_hits, evaluator, loggers


def train_hits(encoder, predictor, data, optimizer, batch_size, 
        mask_target, k_shots=0):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = data.train_pos_edge_index
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    neg_edge_epoch = data.train_neg_edge_index
    assert pos_train_edge.size(1) == neg_edge_epoch.size(1), 'Pos and neg edge index size mismatch'
    # for perm in (pbar := tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
    #                        shuffle=True)) ):
    for perm in tqdm(DataLoader(range(pos_train_edge.size(1)), batch_size,
                           shuffle=True),desc='Train'):
        edge, edge_query_idx, edge_support_labels  = build_QK_graph(perm, data, k_shots, "train", "pos")
        neg_edge, neg_edge_query_idx, neg_edge_support_labels = build_QK_graph(perm, data, k_shots, "train", "neg")
        neg_edge_query_idx  = neg_edge_query_idx + perm.size(0)
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(perm.size()[0]), torch.zeros(perm.size()[0])), dim=0).to(device)
        if mask_target:
            adj_t = data.adj_t
            undirected_edges = torch.cat((train_edges, train_edges.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
        else:
            adj_t = data.adj_t


        h = encoder(data.x, adj_t)
        edge_query_idx = torch.cat((edge_query_idx, neg_edge_query_idx), dim=0)
        edge_support_labels = torch.cat((edge_support_labels, neg_edge_support_labels), dim=0)
        out = predictor(h, adj_t, train_edges, edge_query_idx, edge_support_labels).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        if data.x is not None:
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)
    
    return total_loss / total_examples


@torch.no_grad()
def test_hits(encoder, predictor, data, evaluator, 
         batch_size, use_valedges_as_input, fast_inference, inference_datasets, mask_target, k_shots=0):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    encoding_per_batch = mask_target and (k_shots>0) # mask target only when k_shots>0, when there is positive edges used as support
    if not encoding_per_batch:
        h = encoder(data.x, adj_t)

    def test_split(split, cache_mode=None):
        pos_test_edge = getattr(data, f"{split}_pos_edge_index")
        neg_test_edge = getattr(data, f"{split}_neg_edge_index")

        pos_test_preds = []
        for perm in tqdm(DataLoader(range(pos_test_edge.size(1)), batch_size),desc=f"{split}:pos"):
            edge, edge_query_idx, edge_support_labels  = build_QK_graph(perm, data, k_shots, split, "pos")
            if encoding_per_batch:
                adj_t = data.adj_t
                undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
                target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
                adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
                h = encoder(data.x, adj_t)
            out = predictor(h, adj_t, edge, edge_query_idx, edge_support_labels)
            pos_test_preds += [out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in tqdm(DataLoader(range(neg_test_edge.size(1)), batch_size),desc=f"{split}:neg"):
            neg_edge, neg_edge_query_idx, neg_edge_support_labels = build_QK_graph(perm, data, k_shots, split, "neg")
            if encoding_per_batch:
                adj_t = data.adj_t
                undirected_edges = torch.cat((neg_edge, neg_edge.flip(0)), dim=-1)
                target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
                adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
                h = encoder(data.x, adj_t)
            neg_test_preds += [predictor(h, adj_t, neg_edge, neg_edge_query_idx, neg_edge_support_labels).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        return pos_test_pred, neg_test_pred

    val_pos_pred, val_neg_pred = test_split('val')

    start_time = time.perf_counter()
    if use_valedges_as_input:
        adj_t = data.full_adj_t
        h = encoder(data.x, adj_t)
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode='use'
    else:
        cache_mode=None
    
    test_pos_pred, test_neg_pred = test_split('test', cache_mode=cache_mode)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    if fast_inference:
        # delete cache
        predictor(h, adj_t, None, cache_mode='delete')
    
    results = {}
    if inference_datasets == "#": # return results if no inference_datasets provided
        results1 = evaluation(val_pos_pred, val_neg_pred, test_pos_pred, test_neg_pred, evaluator)
        for key, value in results1.items():
            results[f"{key}#"] = value
        return results
    slice_dict = data._slice_dict
    preds = ()
    for name_prefix in ["val_pos", "val_neg", "test_pos", "test_neg"]:
        # locals()[f"{name_prefix}_edge_index_split"]
        splits = slice_dict[f"{name_prefix}_edge_index"][1:] - slice_dict[f"{name_prefix}_edge_index"][:-1]
        # locals()[f"{name_prefix}_preds"] = torch.split(locals()[f"{name_prefix}_pred"],splits.tolist(),0)
        preds += (torch.split(locals()[f"{name_prefix}_pred"],splits.tolist(),0),)
    
    for i, one_dataset_name in enumerate(inference_datasets.split(',')):
        results1 = evaluation(preds[0][i],preds[1][i],preds[2][i],preds[3][i], evaluator=evaluator)
        for key, value in results1.items():
            results[f"{key}{one_dataset_name}"] = value        

    return results

def evaluation(pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, evaluator):
    results = {}
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))
    return results


def train_mrr(encoder, predictor, data, split_edge, optimizer, batch_size, 
        mask_target, dataset_name, num_neg, adj2):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    source_edge = split_edge['train']['source_node'].to(device)
    target_edge = split_edge['train']['target_node'].to(device)
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True),desc='Train'):
        if mask_target:
            adjmask[perm] = 0
            tei = torch.stack((source_edge[adjmask], target_edge[adjmask]), dim=0) # TODO: check if both direction is removed
            adj_t = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   source_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj_t = adj_t.to_symmetric()
        else:
            adj_t = data.adj_t


        h = encoder(data.x, adj_t)
        dst_neg = torch.randint(0, data.num_nodes, perm.size()*num_neg,
                                dtype=torch.long, device=device)

        edge = torch.stack((source_edge[perm], target_edge[perm]), dim=0)
        neg_edge = torch.stack((source_edge[perm].repeat(num_neg), dst_neg), dim=0)
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(device)
        out = predictor(h, adj_t, train_edges, adj2 = adj2).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        if data.x is not None:
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)
    
    return total_loss / total_examples


@torch.no_grad()
def test_mrr(encoder, predictor, data, split_edge, evaluator, 
         batch_size, use_valedges_as_input, fast_inference, adj2):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    def test_split(split, cache_mode=None):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h, adj_t, torch.stack((src, dst)), cache_mode=cache_mode, adj2 = adj2).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h, adj_t, torch.stack((src, dst_neg)), cache_mode=cache_mode, adj2 = adj2).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return pos_pred, neg_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')

    start_time = time.perf_counter()
    if use_valedges_as_input:
        adj_t = data.full_adj_t
        h = encoder(data.x, adj_t)
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode='use'
    else:
        cache_mode=None
    
    pos_test_pred, neg_test_pred = test_split('test', cache_mode=cache_mode)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    if fast_inference:
        # delete cache
        predictor(h, adj_t, None, cache_mode='delete')
    
    valid_mrr = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })['mrr_list'].mean().item()
    test_mrr = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })['mrr_list'].mean().item()

    results = {
        "MRR": (valid_mrr, test_mrr),
    }

    # Compute AUC
    # neg_valid_pred = neg_valid_pred.view(-1)
    # neg_test_pred = neg_test_pred.view(-1)
    # valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    # valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    # test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    # test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    # results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))

    return results


def build_QK_graph(perm, data, k_shots, split, label):
    assert split in ['train', 'val', 'test']
    assert label in ['pos', 'neg']
    slice_dict = data._slice_dict
    other_label = 'pos' if label == 'neg' else 'neg'
    splits = slice_dict[f"train_{label}_edge_index"][1:] - slice_dict[f"train_{label}_edge_index"][:-1]
    all_edges = getattr(data, f"train_{label}_edge_index")
    other_edges = getattr(data, f"train_{other_label}_edge_index")

    all_mask = torch.ones_like(all_edges[0], dtype=torch.bool)
    if split == "train":
        query_edges = all_edges[:,perm]
        # only remove the overlap edges during training
        all_mask[perm] = False
    else:
        query_edges = getattr(data, f"{split}_{label}_edge_index")[:,perm]
    edges_by_dataset = torch.split(all_edges,splits.tolist(),1)
    other_edges_by_dataset = torch.split(other_edges,splits.tolist(),1)
    masks_by_dataset = torch.split(all_mask,splits.tolist(),0)

    dataset_ids = torch.bucketize(perm, slice_dict[f"{split}_{label}_edge_index"], right=True) - 1
    unique_dataset_ids, inverse_indices, count = torch.unique(dataset_ids, return_inverse=True, return_counts=True)

    # in each dataset, we sample count*k_shots edges with replacement
    # the edges to be sampled are the ones not in the query set
    support_edges = []
    query_idx = []
    support_labels = []
    for i, one_dataset_id in enumerate(unique_dataset_ids):
        edges_to_sample = edges_by_dataset[one_dataset_id][:,masks_by_dataset[one_dataset_id]] # remove the edges in query set
        other_edges_to_sample = other_edges_by_dataset[one_dataset_id]

        assert edges_to_sample.size(1) >= k_shots, 'Less than k_shots edges to be support edges'
        random_indices = torch.randint(0, edges_to_sample.size(1), (count[i]*k_shots,))
        edges_sampled = edges_to_sample[:,random_indices]
        random_indices = torch.randint(0, other_edges_to_sample.size(1), (count[i]*k_shots,))
        other_edges_sampled = other_edges_to_sample[:,random_indices]


        support_edges.append(edges_sampled)
        support_edges.append(other_edges_sampled)

        query_idx1 = torch.where(dataset_ids == one_dataset_id)[0]
        query_idx1 = torch.repeat_interleave(query_idx1, k_shots).repeat(2)

        query_idx.append(query_idx1)
        if label == 'pos':
            support_labels.append(torch.ones_like(edges_sampled[0]))
            support_labels.append(torch.zeros_like(other_edges_sampled[0]))
        else:
            support_labels.append(torch.zeros_like(edges_sampled[0]))
            support_labels.append(torch.ones_like(other_edges_sampled[0]))
    support_edges = torch.cat(support_edges, dim=1)
    query_idx = torch.cat(query_idx, dim=0).to(query_edges.device)
    support_labels = torch.cat(support_labels, dim=0)
    # prepend QUERY_GRAPH_LABEL to support_labels to indicate that they are not support edges
    support_labels = torch.cat((torch.ones_like(query_edges[0]).fill_(QUERY_GRAPH_LABEL), support_labels), dim=0)

    edges = torch.cat((query_edges, support_edges), dim=1)
    # assign the query_idx of query edges to its own index
    query_idx = torch.cat([torch.arange(query_edges.size(1), device=query_edges.device), query_idx], dim=0)
    return edges, query_idx, support_labels


if __name__ == '__main__':
    from utils import merge_data
    from torch_geometric.data import Data

    data1 = Data(edge_index=torch.randint(0, 100, (2, 20)), num_nodes=100)
    data1.train_pos_edge_index = torch.LongTensor([[4,5,6],[1,2,3]])
    data1.train_neg_edge_index = torch.LongTensor([[7,8,9],[10,11,12]])
    data1.val_pos_edge_index = torch.LongTensor([[13,14,15],[16,17,18]])
    data1.val_neg_edge_index = torch.LongTensor([[22,23,24],[19,20,21]])
    data1.test_pos_edge_index = torch.LongTensor([[25,26,27],[28,29,30]])
    data1.test_neg_edge_index = torch.LongTensor([[31,32,33],[34,35,36]])

    data2 = Data(edge_index=torch.randint(0, 100, (2, 20)), num_nodes=100)
    data2.train_pos_edge_index = torch.LongTensor([[36,35,34],[33,32,31]])
    data2.train_neg_edge_index = torch.LongTensor([[30,29,28],[27,26,25]])
    data2.val_pos_edge_index = torch.LongTensor([[24,23,22],[21,20,19]])
    data2.val_neg_edge_index = torch.LongTensor([[18,17,16],[15,14,13]])
    data2.test_pos_edge_index = torch.LongTensor([[12,11,10],[9,8,7]])
    data2.test_neg_edge_index = torch.LongTensor([[6,5,4],[3,2,1]])

    data = merge_data([data1, data2])

    # split= 'train' label = 'pos'
    perm = torch.LongTensor([5, 1])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "train", "pos")
    perm = torch.LongTensor([0, 1])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "train", "pos")
    

    # split= 'train' label = 'neg'
    perm = torch.LongTensor([4, 2])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "train", "neg")
    perm = torch.LongTensor([1, 2])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "train", "neg")

    # split= 'val' label = 'pos'
    perm = torch.LongTensor([3, 0])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "val", "pos")
    perm = torch.LongTensor([3, 4])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "val", "pos")

    # split= 'val' label = 'neg'
    perm = torch.LongTensor([3, 0])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "val", "neg")
    perm = torch.LongTensor([4, 5])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "val", "neg")

    # split= 'test' label = 'pos'
    perm = torch.LongTensor([1, 4])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "test", "pos")
    perm = torch.LongTensor([4, 5])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "test", "pos")

    # split= 'test' label = 'neg'
    perm = torch.LongTensor([1, 4])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "test", "neg")
    perm = torch.LongTensor([0, 2])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "test", "neg")


    # test build_QK_edge_index
    from models import build_QK_edge_index, AttentionLayer
    in_channels = 1
    out_channels = 4
    gat = AttentionLayer(in_channels=in_channels, out_channels=out_channels, 
                                  heads=4, concat=False, add_self_loops=False,
                                  use_graph_embedding=True)

    
    embedding = nn.Embedding(QUERY_GRAPH_LABEL+1, out_channels)
    perm = torch.LongTensor([1, 4])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "test", "neg")
    x = torch.arange(query_idx.size(0)).view(-1,1).float() # in_channels = 1
    data_graph_x, data_graph_edge_index, T_embedding, query_graph_mask = build_QK_edge_index(
                x, query_idx, support_labels, embedding, True)
    
    perm = torch.LongTensor([4, 5])
    edges, query_idx, support_labels = build_QK_graph(perm, data, 1, "test", "pos")
    x = torch.arange(query_idx.size(0)).view(-1,1).float()
    data_graph_x, data_graph_edge_index, T_embedding, query_graph_mask = build_QK_edge_index(
                x, query_idx, support_labels, embedding, False)

    x_new, (_,alpha) = gat(data_graph_x, data_graph_edge_index, T_embedding, return_attention_weights=True)
