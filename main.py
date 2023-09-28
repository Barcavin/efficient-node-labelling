import argparse
import os
import sys
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
from node_label import spmdiff_
from utils import ( get_dataset, data_summary,
                   set_random_seeds, str2bool, get_data_split, initial_embedding)


def train(encoder, predictor, data, split_edge, optimizer, batch_size, 
        mask_target, dataset_name):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    source_edge = split_edge['train']['source_node'].to(device)
    target_edge = split_edge['train']['target_node'].to(device)
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    # for perm in (pbar := tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
    #                        shuffle=True)) ):
    for perm in DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True):
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


        # if dataset != "collab" and dataset != "ppa":
        # neg_edge = negative_sampling(data.edge_index, num_nodes=create_input(data).size(0),
        #                         num_neg_samples=perm.size(0), method='sparse')
        # elif dataset == "collab" or dataset == "ppa":
        #     neg_edge = torch.randint(0, create_input(data).size()[0], edge.size(), dtype=torch.long,
        #                      device=device)
        dst_neg = torch.randint(0, data.num_nodes, perm.size(),
                                dtype=torch.long, device=device)
        edge = torch.stack((source_edge[perm], target_edge[perm]), dim=0)
        neg_edge = torch.stack((source_edge[perm], dst_neg), dim=0)
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(device)
        out = predictor(h, adj_t, train_edges).squeeze()
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
def test(encoder, predictor, data, split_edge, evaluator, 
         batch_size, use_valedges_as_input, fast_inference):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.full_adj_t
    h = encoder(data.x, adj_t)
    def test_split(split):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h, adj_t, torch.stack((src, dst))).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h, adj_t, torch.stack((src, dst_neg))).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = 0.0 #test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return {"MRR":(valid_mrr, test_mrr)}

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    # dataset setting
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    # model setting
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--xdp', type=float, default=0.2)
    parser.add_argument('--feat_dropout', type=float, default=0.5)
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dothash_dim', type=int, default=1024)
    parser.add_argument('--minimum_degree_onehot', type=int, default=-1, help='minimum degree for onehot encoding during dothash to reduce variance')
    parser.add_argument('--predictor', type=str, default='mlp', choices=["inner","mlp","ENL","DP+exact","DP+prop_only","DP+combine"])  ##inner/mlp
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--feature_combine', type=str, default='hadamard', choices=['hadamard','plus_minus'], help='how to represent a link with two nodes features')
    parser.add_argument('--jk', type=str2bool, default='True', help='whether to use Jumping Knowledge')
    parser.add_argument('--batchnorm_affine', type=str2bool, default='True', help='whether to use Affine in BatchNorm')
    parser.add_argument('--use_embedding', type=str2bool, default='False', help='whether to train node embedding')
    parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges when computing node labelling')
    parser.add_argument('--dgcnn', type=str2bool, default='False', help='whether to use DGCNN as the target edge pooling')
    parser.add_argument('--torchhd_style', type=str2bool, default='True', help='whether to use torchhd to randomize vectors')
    parser.add_argument('--use_degree', type=str, default='none', choices=["none","mlp","AA","RA"], help="the way to encode node weights")
    parser.add_argument('--fast_inference', type=str2bool, default='False', help='whether to enable a faster inference by caching the node vectors')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='MRR', help='main evaluation metric')

    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()
    # start time
    start_time = time.time()
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # device = torch.device('cpu')

    data, split_edge = get_dataset(args.dataset_dir, args.dataset, args.use_valedges_as_input, args.year)
    if data.x is None:
        args.use_feature = False

    if args.print_summary:
        data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    else:
        print(args)
    final_log_path = Path(args.log_dir) / f"{args.dataset}_jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}.log"
    with open(final_log_path, 'w') as f:
        print(args, file=f)
    
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(final_log_path, 'a') as f:
        f.write('\n' + cmd_input)
    
    evaluator = Evaluator(name=args.dataset)
    loggers = {
        'MRR': Logger(args.runs, args)
    }

    val_max = 0.0
    for run in range(args.runs):
        if not args.dataset.startswith('ogbl-'):
            data, split_edge = get_data_split(args.dataset_dir, args.dataset, args.val_ratio, args.test_ratio, run=run)
            data = T.ToSparseTensor(remove_edge_index=False)(data)
            # Use training + validation edges for inference on test set.
            if args.use_valedges_as_input:
                val_edge_index = split_edge['valid']['edge'].t()
                full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
                data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
                                                        sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
                data.full_adj_t = data.full_adj_t.to_symmetric()
            else:
                data.full_adj_t = data.adj_t
            if args.data_split_only:
                if run == args.runs - 1:
                    exit(0)
                else:
                    continue
        
        if args.minimum_degree_onehot > 0:
            d_v = degree(data.edge_index[0],data.num_nodes)
            nodes_to_one_hot = d_v >= args.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            print(f"number of nodes to onehot: {int(one_hot_dim)}")
        data = data.to(device)
        if args.use_embedding:
            emb = initial_embedding(data, args.hidden_channels, device)
        else:
            emb = None
        if 'gcn' in args.encoder:
            encoder = GCN(data.num_features, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.feat_dropout, args.xdp, args.use_feature, args.jk, args.encoder, emb).to(device)
        elif args.encoder == 'sage':
            encoder = SAGE(data.num_features, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.feat_dropout, args.xdp, args.use_feature, args.jk, emb).to(device)
        elif args.encoder == 'appnp':
            encoder = APPNP_model(input_size, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.dropout).to(device)
        elif args.encoder == 'gat':
            encoder = GAT(input_size, args.hidden_channels,
                        args.hidden_channels, 1,
                        args.dropout).to(device)
        elif args.encoder == 'mlp':
            encoder = MLP(args.num_layers, input_size, 
                          args.hidden_channels, args.hidden_channels, args.dropout).to(device)

        predictor_in_dim = args.hidden_channels * int(args.use_feature or args.use_embedding)
                            # * (1 + args.jk * (args.num_layers - 1))
        if args.predictor in ['inner','mlp']:
            predictor = LinkPredictor(args.predictor, predictor_in_dim, args.hidden_channels, 1,
                                    args.num_layers, args.feat_dropout).to(device)
        elif args.predictor == 'ENL':
            predictor = EfficientNodeLabelling(predictor_in_dim, args.hidden_channels,
                                    args.num_layers, args.feat_dropout, args.num_hops, 
                                    dgcnn=args.dgcnn, use_degree=args.use_degree).to(device)
        elif 'DP' in args.predictor:
            prop_type = args.predictor.split("+")[1]
            predictor = DotProductLabelling(predictor_in_dim, args.hidden_channels,
                                    args.num_layers, args.feat_dropout, args.label_dropout, args.num_hops, 
                                    prop_type=prop_type, torchhd_style=args.torchhd_style,
                                    use_degree=args.use_degree, dothash_dim=args.dothash_dim,
                                    minimum_degree_onehot=args.minimum_degree_onehot, batchnorm_affine=args.batchnorm_affine,
                                    feature_combine=args.feature_combine).to(device)

        encoder.reset_parameters()
        predictor.reset_parameters()
        parameters = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, data, split_edge,
                         optimizer, args.batch_size, args.mask_target, args.dataset)

            results = test(encoder, predictor, data, split_edge,
                            evaluator, args.batch_size, args.use_valedges_as_input, args.fast_inference)

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
                    to_print = (f'Run: {run + 1:02d}, ' +
                            f'Epoch: {epoch:02d}, '+
                            f'Loss: {loss:.4f}, '+
                            f'Valid: {100 * valid_hits:.2f}%, '+
                            f'Test: {100 * test_hits:.2f}%')
                    print(key)
                    print(to_print)
                    with open(final_log_path, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
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
