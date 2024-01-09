import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_undirected
from torch_sparse import SparseTensor
from tqdm import tqdm

from train_utils import get_train_test
from models import GCN, MLP, SAGE, LinkPredictor, MPLP
from utils import ( data_summary, get_pretrain_data, get_inference_data, save_model, load_model,
                   set_random_seeds, str2bool, get_git_revision_short_hash, update_args)

MPLP_dict={
    "MPLP": "combine",
    "MPLP+": "prop_only",
    "MPLP+exact": "exact",
    "MPLP+precompute": "precompute",
}

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    # dataset setting
    parser.add_argument('--pretrain_datasets', type=str, default="Router,Yeast,Ecoli,Power,PolBlogs,Physics,Citeseer,Pubmed,musae-twitch,musae-github")
    parser.add_argument('--inference_datasets', type=str, default='Celegans,USAir,PB,NS,Cora,CS,Photo,musae-facebook,syn-TRIANGULAR,syn-GRID')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    # MPLP settings
    parser.add_argument('--signature_dim', type=int, default=1024, help="the node signature dimension `F` in MPLP")
    parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges to remove the shortcut')
    parser.add_argument('--use_degree', type=str, default='none', choices=["none","mlp","AA","RA"], help="rescale vector norm to facilitate weighted count")
    parser.add_argument('--signature_sampling', type=str, default='torchhd', help='whether to use torchhd to randomize vectors', choices=["torchhd","gaussian","onehot"])
    parser.add_argument('--fast_inference', type=str2bool, default='False', help='whether to enable a faster inference by caching the node vectors')
    
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

    # training setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@50#', help='main evaluation metric')
    parser.add_argument('--save_model', type=str, default=None, help='the directory to save model')
    parser.add_argument('--load_model', type=str, default=None, help='the directory to load model')

    # in-context setting
    parser.add_argument('--train_samples', type=int, default=None, help='number of training samples per pretrain dataset')
    parser.add_argument('--k_shots', type=int, default=0, help='number of pos/neg support edges per sample')
    parser.add_argument('--test_k_shots', type=int, default=None, help='number of pos/neg support edges per sample')
    parser.add_argument('--foundation_mode', type=str2bool, default=False, help='whether to use in-context training')
    parser.add_argument('--heads', type=int, default=4, help='number of heads in MPLP')
    parser.add_argument('--add_self_loops', type=str2bool, default=True, help='whether to add self loops in MPLP')
    parser.add_argument('--use_graph_embedding', type=str2bool, default=True, help='whether to use subgraph embedding')


    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()
    # start time
    start_time = time.time()
    set_random_seeds(234)

    if args.test_k_shots is None:
        args.test_k_shots = args.k_shots

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    device = torch.device(device)
    # device = torch.device('cpu')

    # backward compatibility
    args.use_feature = False
    predictor_in_dim = 0
    args.minimum_degree_onehot = -1
    args.adj2 = False

    # if args.print_summary:
    #     data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    # else:
    print(args)
    appendix = f"jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}"
    final_log_path = Path(args.log_dir) / f"Pretrain_{appendix}.log"
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print('Command line input: ' + cmd_input + ' is saved.')
    # Save git revision.
    git_hash = get_git_revision_short_hash()
    hostname = os.getenv('HOSTNAME','None')
    final_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_log_path, 'w') as f:
        print(args, file=f)
        f.write('\n' + cmd_input)
        print(f"HOSTNAME: {hostname}", file=f)
        print('Git revision: ' + git_hash + '\n', file=f)
    
    train, test, evaluator, loggers = get_train_test(args)

    train_data = get_pretrain_data(args.dataset_dir, args.pretrain_datasets, args.train_samples).to(device)
    train_data_num_features = train_data.num_features
    val_max = 0.0
    test_data = get_inference_data(args.dataset_dir, args.inference_datasets, 0).to(device)
    # override args when loading model
    if args.load_model:
        train_data_num_features = 0
        args = update_args(args, args.load_model)
        args_str = "###### Override args ######\n" + args.__repr__()
        print(args_str)
        with open(final_log_path, 'a') as f:
            print(args_str, file=f)
        
    if 'gcn' in args.encoder:
        encoder = GCN(train_data_num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.feat_dropout, args.xdp, args.use_feature, args.jk, args.encoder).to(device)
    elif args.encoder == 'sage':
        encoder = SAGE(train_data_num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.feat_dropout, args.xdp, args.use_feature, args.jk).to(device)
    elif args.encoder == 'mlp':
        encoder = MLP(args.num_layers, train_data_num_features, 
                        args.hidden_channels, args.hidden_channels, args.dropout).to(device)
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
                        feature_combine=args.feature_combine, adj2=args.adj2, add_self_loops=args.add_self_loops, foundation_mode=args.foundation_mode, heads=args.heads, use_graph_embedding=args.use_graph_embedding)
        if prop_type == "precompute":
            assert args.use_degree != "mlp"
            predictor.precompute(train_data.adj_t)
    predictor = predictor.to(device)

    encoder.reset_parameters()
    predictor.reset_parameters()
    parameters = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')

    cnt_wait = 0
    best_val = 0.0
    run = 0
    state_dict = None
    for epoch in range(1, 1 + args.epochs):
        if args.load_model:
            # no need to train
            break
        loss = train(encoder, predictor, train_data,
                        optimizer, args.batch_size, args.mask_target, args.k_shots)

        results = test(encoder, predictor, test_data,
                        evaluator, args.test_batch_size, args.use_valedges_as_input, args.fast_inference, "#", args.test_k_shots)

        if results[args.metric][0] >= best_val:
            best_val = results[args.metric][0]
            cnt_wait = 0
            state_dict = copy.deepcopy({
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
            })
            if args.save_model:
                # save best model
                save_model(state_dict, args.save_model, appendix,
                        cmd_input, git_hash, hostname, args)
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
    
    to_pops = []
    for key in loggers.keys():
        if key[-1] == '#': # pop validation
            to_pops.append(key)
    for key in to_pops:
        loggers.pop(key)

    # load best model
    load_model(encoder, predictor, args.load_model, state_dict)
    for run in range(args.runs):
        test_data = get_inference_data(args.dataset_dir, args.inference_datasets, run).to(device)
        results = test(encoder, predictor, test_data,
                        evaluator, args.test_batch_size, args.use_valedges_as_input, args.fast_inference, args.inference_datasets, args.k_shots)
        for key, result in results.items():
            loggers[key].add_result(run, result)
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
