import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_sparse import SparseTensor

from train_utils import get_train_test
from models import GCN, MLP, SAGE, LinkPredictor, MPLP
from node_label import get_two_hop_adj
from utils import ( get_dataset, data_summary, get_git_revision_short_hash,
                   set_random_seeds, str2bool, get_data_split, initial_embedding)

MPLP_dict={
    "MPLP": "combine",
    "MPLP+": "prop_only",
    "MPLP+exact": "exact",
    "MPLP+precompute": "precompute",
}

def load_model(encoder, predictor, load_model_path, state_dict=None):
    if load_model_path is not None:
        state_dict = torch.load(load_model_path)
    encoder.load_state_dict(state_dict['encoder'])
    predictor.load_state_dict(state_dict['predictor'])
    print(f"load model from {load_model_path}")

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    # dataset setting
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    # MPLP settings
    parser.add_argument('--signature_dim', type=int, default=1024, help="the node signature dimension `F` in MPLP")
    parser.add_argument('--minimum_degree_onehot', type=int, default=-1, help='the minimum degree of hubs with onehot encoding to reduce variance')
    parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges to remove the shortcut')
    parser.add_argument('--use_degree', type=str, default='none', choices=["none","mlp","AA","RA"], help="rescale vector norm to facilitate weighted count")
    parser.add_argument('--signature_sampling', type=str, default='torchhd', help='whether to use torchhd to randomize vectors', choices=["torchhd","gaussian","onehot"])
    parser.add_argument('--fast_inference', type=str2bool, default='False', help='whether to enable a faster inference by caching the node vectors')
    parser.add_argument('--adj2', type=str2bool, default="False", help='Whether to use 2-hop adj for MPLP+prop_only.')
    
    # model setting
    parser.add_argument('--predictor', type=str, default='MPLP', choices=["inner","mlp","ENL",
    "MPLP+exact","MPLP+","MPLP","MPLP+precompute"])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--xdp', type=float, default=0.2)
    parser.add_argument('--feat_dropout', type=float, default=0.5)
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--feature_combine', type=str, default='hadamard', choices=['hadamard','plus_minus'], help='how to represent a link with two nodes features')
    parser.add_argument('--jk', type=str2bool, default='True', help='whether to use Jumping Knowledge')
    parser.add_argument('--batchnorm_affine', type=str2bool, default='True', help='whether to use Affine in BatchNorm')
    parser.add_argument('--use_embedding', type=str2bool, default='False', help='whether to train node embedding')
    # parser.add_argument('--dgcnn', type=str2bool, default='False', help='whether to use DGCNN as the target edge pooling')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')

    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--train_samples', type=int, default=None, help='number of training samples per pretrain dataset')

    args = parser.parse_args()
    # start time
    start_time = time.time()
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # device = torch.device('cpu')

    data, split_edge = get_dataset(args.dataset_dir, args.dataset, args.use_valedges_as_input, args.year)
    if args.dataset == "ogbl-citation2":
        args.metric = "MRR"
    if data.x is None:
        args.use_feature = False

    if args.print_summary:
        data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    else:
        print(args)
    
    final_log_path = Path(args.log_dir) / f"{args.dataset}_jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}.log"    
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print('Command line input: ' + cmd_input + ' is saved.')
    # Save git revision.
    git_hash = get_git_revision_short_hash()
    final_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_log_path, 'w') as f:
        print(args, file=f)
        f.write('\n' + cmd_input)
        print(f"HOSTNAME: {os.getenv('HOSTNAME','None')}", file=f)
        print('Git revision: ' + git_hash + '\n', file=f)
    
    train, test, evaluator, loggers = get_train_test(args)

    val_max = 0.0
    for run in range(args.runs):
        if not args.dataset.startswith('ogbl-'):
            data, split_edge = get_data_split(args.dataset_dir, args.dataset, args.val_ratio, args.test_ratio, run=run)
            split_edge['train']['edge'] = split_edge['train']['edge'][:args.train_samples]
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
        if args.adj2:
            print("Computing 2-hop adj. This may take a while.")
            start_time = time.time()
            adj_t_no_value = data.adj_t.clone().set_value_(None)
            _, adj2 = get_two_hop_adj(adj_t_no_value)
            end_time = time.time()
            print(f"Computing 2-hop adj took {end_time - start_time:.4f}s")
        else:
            adj2 = None
        if args.minimum_degree_onehot > 0:
            d_v = data.adj_t.sum(dim=0).to_dense()
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
        elif args.encoder == 'mlp':
            encoder = MLP(args.num_layers, data.num_features, 
                          args.hidden_channels, args.hidden_channels, args.dropout).to(device)

        predictor_in_dim = args.hidden_channels * int(args.use_feature or args.use_embedding)
                            # * (1 + args.jk * (args.num_layers - 1))
        if args.predictor in ['inner','mlp']:
            predictor = LinkPredictor(args.predictor, predictor_in_dim, args.hidden_channels, 1,
                                    args.num_layers, args.feat_dropout)
        # elif args.predictor == 'ENL':
        #     predictor = NaiveNodeLabelling(predictor_in_dim, args.hidden_channels,
        #                             args.num_layers, args.feat_dropout, args.num_hops, 
        #                             dgcnn=args.dgcnn, use_degree=args.use_degree).to(device)
        elif 'MPLP' in args.predictor:
            prop_type = MPLP_dict[args.predictor]
            predictor = MPLP(predictor_in_dim, args.hidden_channels,
                                    args.num_layers, args.feat_dropout, args.label_dropout, args.num_hops, 
                                    prop_type=prop_type, signature_sampling=args.signature_sampling,
                                    use_degree=args.use_degree, signature_dim=args.signature_dim,
                                    minimum_degree_onehot=args.minimum_degree_onehot, batchnorm_affine=args.batchnorm_affine,
                                    feature_combine=args.feature_combine, adj2=args.adj2)
            if prop_type == "precompute":
                assert args.use_degree != "mlp"
                predictor.precompute(data.adj_t)
        predictor = predictor.to(device)

        encoder.reset_parameters()
        predictor.reset_parameters()
        parameters = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')

        cnt_wait = 0
        best_val = 0.0
        # load best model
        load_model(encoder, predictor, args.load_model, None)

        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, data, split_edge,
                         optimizer, args.batch_size, args.mask_target, args.dataset, 
                         num_neg=args.num_neg, adj2=adj2)

            results = test(encoder, predictor, data, split_edge,
                            evaluator, args.test_batch_size, args.use_valedges_as_input, args.fast_inference, adj2=adj2)

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
            # import json
            # with open("threshold.json") as f:
            #     threshold = json.load(f).get(args.dataset,[]) # threshold is a list [(epoch, performance)]
            # for t_epoch, t_value in threshold:
            #     if epoch >= t_epoch and results[args.metric][1]*100 < t_value:
            #         print(f"Discard due to low test performance {results[args.metric][1]} < {t_value} after epoch {t_epoch}")
            #         break
            # else:
            #     continue
            # break

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
