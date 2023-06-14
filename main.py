import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from sklearn.metrics import *
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch_geometric.utils import (degree,
                                   negative_sampling)
from tqdm import tqdm

from discrepancy import cmd
from logger import Logger
from models import GAT, GCN, MLP, SAGE, APPNP_model, LinkPredictor
from utils import ( get_dataset, data_summary, initialize, create_input,
                   set_random_seeds, str2bool, get_data_split, make_edge_index)


def train(model, predictor, data, split_edge, optimizer, batch_size, encoder_name, dataset):
    model.train()
    predictor.train()

    criterion = BCELoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(create_input(data).device)
    edge_index = data.edge_index
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    # for perm in (pbar := tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
    #                        shuffle=True)) ):
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        if encoder_name == 'mlp':
            h = model(create_input(data))
        else:
            h = model(create_input(data), edge_index)

        edge = pos_train_edge[perm].t()

        if dataset != "collab" and dataset != "ppa":
            neg_edge = negative_sampling(data.full_edge_index, num_nodes=create_input(data).size(0),
                                 num_neg_samples=perm.size(0), method='dense')
        elif dataset == "collab" or dataset == "ppa":
            neg_edge = torch.randint(0, create_input(data).size()[0], edge.size(), dtype=torch.long,
                             device=h.device)

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(create_input(data), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
    
    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        h = model(create_input(data))
    else:
        h = model(create_input(data), data.edge_index)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_valid_preds = []
    pos_valid_x = []
    pos_valid_out = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        out, x = predictor(h[edge[0]], h[edge[1]],return_hidden=True)
        pos_valid_preds += [out.squeeze().cpu()]
        pos_valid_x += [x.cpu()]
        pos_valid_out += [out.cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    pos_valid_x = torch.cat(pos_valid_x, dim=0)
    pos_valid_out = torch.cat(pos_valid_out, dim=0).squeeze()

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    pos_test_x = []
    pos_test_out = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        out, x = predictor(h[edge[0]], h[edge[1]],return_hidden=True)
        pos_test_preds += [out.squeeze().cpu()]
        pos_test_x += [x.cpu()]
        pos_test_out += [out.cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    pos_test_x = torch.cat(pos_test_x, dim=0)
    pos_test_out = torch.cat(pos_test_out, dim=0).squeeze()

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    if dataset != "collab" and dataset != "ppa":
        for K in [10, 20, 30, 50]:
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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--datasets', type=str, default='collab')
    parser.add_argument('--initial', type=str, default='', choices=['', 'one-hot', 'trainable'])
    parser.add_argument('--predictor', type=str, default='mlp')  ##inner/mlp
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()
    if not args.print_summary:
        print(args)
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # if args.datasets == "cora" or args.datasets == "citeseer" or args.datasets == "pubmed":
    if args.datasets != "collab" and args.datasets != "ppa":
        dataset = get_dataset(args.dataset_dir, args.datasets)
        data = dataset[0]

    elif args.datasets == "collab" or args.datasets == "ppa":
        dataset = PygLinkPropPredDataset(name=('ogbl-' + args.datasets), root=args.dataset_dir)
        data = dataset[0]
        # data = T.ToSparseTensor()(data)

        split_edge = dataset.get_edge_split()
        print("-"*20)
        print(f"train: {split_edge['train']['edge'].shape[0]}")
        print(f"{split_edge['train']['edge'][:10,:]}")
        print(f"valid: {split_edge['valid']['edge'].shape[0]}")
        print(f"test: {split_edge['test']['edge'].shape[0]}")
        print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
        data.edge_index = make_edge_index(split_edge["train"]["edge"])
        input_size = data.num_features
        # args.metric = 'Hits@50'
        ##add training edges into message passing
        # data.adj_t = torch.cat((edge_index, split_edge['train']['edge'].t()), dim=1)
        # split_edge['train']['edge'] = data.adj_t.t()

    if args.print_summary:
        data_summary(args.datasets, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    final_log_path = Path(args.log_dir) / f"{args.datasets}_{args.encoder}_val_{int(100*args.val_ratio)}_test_{int(100*args.test_ratio)}_{args.initial}_{int(time.time())}.txt"
    with open(final_log_path, 'w') as f:
        print(args, file=f)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    if args.datasets != "collab" and args.datasets != "ppa":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@30': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }
    elif args.datasets == "collab" or args.datasets == "ppa":
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }

    val_max = 0.0
    for run in range(args.runs):
        if args.datasets != "collab" and args.datasets != "ppa":
            data, split_edge = get_data_split(args.dataset_dir, args.datasets, args.val_ratio, args.test_ratio, run=run)
            if args.data_split_only:
                if run == args.runs - 1:
                    exit(0)
                else:
                    continue
        data, input_size = initialize(data, args.initial)
        data = data.to(device)
        if args.encoder == 'sage':
            model = SAGE(args.datasets, input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'gcn':
            model = GCN(input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'appnp':
            model = APPNP_model(input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'gat':
            model = GAT(input_size, args.hidden_channels,
                        args.hidden_channels, 1,
                        args.dropout).to(device)
        elif args.encoder == 'mlp':
            model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

        model.reset_parameters()
        predictor.reset_parameters()
        parameters = list(model.parameters()) + list(predictor.parameters())
        if hasattr(data, "emb"):
            parameters += list(data.emb.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr)

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge,
                         optimizer, args.batch_size, args.encoder, args.datasets)

            results = test(model, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.datasets)

            # if results['Hits@50'][1] > val_max:
            #     val_max = results['Hits@50'][1]
            #     if args.encoder == 'sage':
            #         torch.save({'gnn': model.state_dict(), 'predictor': predictor.state_dict()}, "saved_models/" + args.datasets + "-sage.pkl")
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


if __name__ == "__main__":
    main()
