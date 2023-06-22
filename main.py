import argparse
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch_geometric.utils import (degree,
                                   negative_sampling)
from torch_sparse import SparseTensor
from tqdm import tqdm

from logger import Logger
from models import GAT, GCN, MLP, SAGE, APPNP_model, LinkPredictor, EfficientNodeLabelling, DotProductLabelling
from node_label import spmnotoverlap_
from utils import ( get_dataset, data_summary, initialize, create_input,
                   set_random_seeds, str2bool, get_data_split, make_edge_index)


def train(encoder, predictor, data, split_edge, optimizer, batch_size, encoder_name, 
          dataset, use_sp_matrix, mask_target):
    encoder.train()
    predictor.train()

    criterion = BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(create_input(data).device)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    # for perm in (pbar := tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
    #                        shuffle=True)) ):
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        edge = pos_train_edge[perm].t()
        if mask_target:
            adj_t = data.adj_t
            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t, _ = spmnotoverlap_(adj_t, target_adj)
        else:
            adj_t = data.adj_t


        if encoder_name == 'mlp' or isinstance(encoder, nn.Identity):
            h = encoder(create_input(data))
        elif use_sp_matrix:
            h = encoder(create_input(data), adj_t)
        else:
            h = encoder(create_input(data), data.edge_index)


        if dataset != "collab" and dataset != "ppa":
            neg_edge = negative_sampling(data.edge_index, num_nodes=create_input(data).size(0),
                                 num_neg_samples=perm.size(0), method='dense')
        elif dataset == "collab" or dataset == "ppa":
            neg_edge = torch.randint(0, create_input(data).size()[0], edge.size(), dtype=torch.long,
                             device=create_input(data).device)

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(create_input(data).device)
        out = predictor(h, adj_t, train_edges).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(create_input(data), 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)
    
    return total_loss / total_examples


@torch.no_grad()
def test(encoder, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset, use_sp_matrix):
    encoder.eval()
    predictor.eval()

    if encoder_name == 'mlp' or isinstance(encoder, nn.Identity):
        h = encoder(create_input(data))
    elif use_sp_matrix:
        h = encoder(create_input(data), data.adj_t)
    else:
        h = encoder(create_input(data), data.edge_index)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        out = predictor(h, data.adj_t, edge)
        pos_valid_preds += [out.squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, data.adj_t, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        out = predictor(h, data.adj_t, edge)
        pos_test_preds += [out.squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, data.adj_t, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    if dataset != "collab" and dataset != "ppa":
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
    elif dataset == "collab" or dataset == "ppa":
        for K in [10, 50, 100]:
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

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    # dataset setting
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--initial', type=str, default='trainable', choices=['', 'one-hot', 'trainable'])
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')

    # model setting
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--predictor', type=str, default='mlp', choices=["inner","mlp","ENL","DP+exact","DP+prop_only","DP+combine"])  ##inner/mlp
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges when computing node labelling')
    parser.add_argument('--use_sp_matrix', type=str2bool, default='True', help='use sparse matrix for adjacency matrix')
    parser.add_argument('--dgcnn', type=str2bool, default='False', help='whether to use DGCNN as the target edge pooling')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')

    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()
    # start time
    start_time = time.time()
    if args.mask_target and not args.use_sp_matrix:
        raise ValueError('mask_target can only be used when use_sp_matrix is True')
    if not args.print_summary:
        print(args)
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # if args.dataset == "cora" or args.dataset == "citeseer" or args.dataset == "pubmed":
    if args.dataset != "collab" and args.dataset != "ppa":
        dataset = get_dataset(args.dataset_dir, args.dataset)
        data = dataset[0]

    elif args.dataset == "collab" or args.dataset == "ppa":
        dataset = PygLinkPropPredDataset(name=('ogbl-' + args.dataset), root=args.dataset_dir)
        data = dataset[0]

        split_edge = dataset.get_edge_split()
        print("-"*20)
        print(f"train: {split_edge['train']['edge'].shape[0]}")
        print(f"{split_edge['train']['edge'][:10,:]}")
        print(f"valid: {split_edge['valid']['edge'].shape[0]}")
        print(f"test: {split_edge['test']['edge'].shape[0]}")
        print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
        data.edge_index = make_edge_index(split_edge["train"]["edge"])
        input_size = data.num_features
        if args.use_sp_matrix:
            data = T.ToSparseTensor(remove_edge_index=False)(data)
        # args.metric = 'Hits@50'
        ##add training edges into message passing
        # data.adj_t = torch.cat((edge_index, split_edge['train']['edge'].t()), dim=1)
        # split_edge['train']['edge'] = data.adj_t.t()

    if args.print_summary:
        data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    final_log_path = Path(args.log_dir) / f"{args.dataset}_jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}.log"
    with open(final_log_path, 'w') as f:
        print(args, file=f)

    if args.predictor in ['inner','mlp']:
        predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                                args.num_layers, args.dropout).to(device)
    elif args.predictor == 'ENL':
        predictor = EfficientNodeLabelling(args.hidden_channels, args.hidden_channels,
                                args.num_layers, args.dropout, args.num_hops, dgcnn=args.dgcnn, use_feature=args.use_feature).to(device)
    elif 'DP' in args.predictor:
        prop_type = args.predictor.split("+")[1]
        predictor = DotProductLabelling(args.hidden_channels, args.hidden_channels,
                                args.num_layers, args.dropout, args.num_hops, use_feature=args.use_feature, prop_type=prop_type).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    if args.dataset != "collab" and args.dataset != "ppa":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }
    elif args.dataset == "collab" or args.dataset == "ppa":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }

    val_max = 0.0
    for run in range(args.runs):
        if args.dataset != "collab" and args.dataset != "ppa":
            data, split_edge = get_data_split(args.dataset_dir, args.dataset, args.val_ratio, args.test_ratio, run=run)
            if args.use_sp_matrix:
                data = T.ToSparseTensor(remove_edge_index=False)(data)
            if args.data_split_only:
                if run == args.runs - 1:
                    exit(0)
                else:
                    continue
        data, input_size = initialize(data, args.initial)
        data = data.to(device)
        if not args.use_feature:
            # not using node features
            encoder = nn.Identity()
        elif args.encoder == 'sage':
            encoder = SAGE(args.dataset, input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'gcn':
            encoder = GCN(input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'appnp':
            encoder = APPNP_model(input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'gat':
            encoder = GAT(input_size, args.hidden_channels,
                        args.hidden_channels, 1,
                        args.dropout).to(device)
        elif args.encoder == 'mlp':
            encoder = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

        if args.use_feature:
            encoder.reset_parameters()
        predictor.reset_parameters()
        parameters = list(encoder.parameters()) + list(predictor.parameters())
        if hasattr(data, "emb") and args.use_feature:
            parameters += list(data.emb.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, data, split_edge,
                         optimizer, args.batch_size, args.encoder, args.dataset, 
                         args.use_sp_matrix, args.mask_target)

            results = test(encoder, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.dataset, args.use_sp_matrix)

            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(final_log_path, 'a') as f:
                print(key,file=f)
                loggers[key].print_statistics(run=run, file=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(final_log_path, 'a') as f:
            print(key,file=f)
            loggers[key].print_statistics(file=f)
    # end time
    end_time = time.time()
    with open(final_log_path, 'a') as f:
        print(f"Total time: {end_time - start_time:.4f}s", file=f)

if __name__ == "__main__":
    main()
