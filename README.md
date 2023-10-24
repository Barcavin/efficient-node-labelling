# Code for Message Passing Link Predictor (MPLP)
Paper: [Pure Message Passing Can Estimate Common Neighbor for Link Prediction](https://arxiv.org/abs/2309.00976)

Authors: Kaiwen Dong, Zhichun Guo, Nitesh V. Chawla

![Framework of MPLP](misc/diagram-framework.png)

## Introduction
This repository contains the code for the paper [Pure Message Passing Can Estimate Common Neighbor for Link Prediction](https://arxiv.org/abs/2309.00976). MPLP is a simple yet effective message passing framework for link prediction. It is based on the observation that the common neighbor count can be estimated by performing message passing on the random vectors. MPLP is able to achieve state-of-the-art performance on a wide range of datasets on various domains, including social networks, biological networks, and citation networks.


## Environment Setting
```
conda create -n mplp
conda install -n mplp pytorch torchvision torchaudio pytorch-cuda=11.8 pyg=2.3.0 pytorch-sparse=0.6.17 -c pytorch -c nvidia -c pyg
conda activate mplp
pip install ogb torch-hd
```

## Data Preparation
Part of the data has been included in the repository at `./data/`. For the rest of the data, it will be automatically downloaded by the code.

## Usage

To run experiments:
```
python main.py --dataset=physics --batch_size=2048 --use_degree=mlp --minimum_degree_onehot=60 --mask_target=True
```

In MPLP, there are couple of hyperparameters that can be tuned, including:

- `--dataset`: the name of the dataset to be used.
- `--batch_size`: the batch size.
- `--signature_dim`: the node signature dimension `F` in MPLP
- `--mask_target`: whether to mask the target node in the training set to remove the shortcut.
- `--use_degree`: the methods to rescale the norm of random vectors.
- `--minimum_degree_onehot`: the minimum degree of hubs with onehot encoding to reduce variance.


## Experiment Reproduction

<details>
<summary>Commands to reproduce the results reported in the paper</summary>

### USAir
```
python main.py --dataset=USAir --xdp=0.8 --feat_dropout=0.2 --use_embedding=True --batch_size=512 --weight_decay=0.001 --use_degree=RA --lr=0.01 --batchnorm_affine=False
```
### NS
```
python main.py --dataset=NS --label_dropout=0.05 --use_feature=False --batch_size=512 --weight_decay=0.001 --use_degree=AA
```
### PB
```
python main.py --dataset=PB --label_dropout=0.05 --use_feature=False --use_embedding=True --batch_size=1024 --minimum_degree_onehot=100
```
### Yeast
```
python main.py --dataset=Yeast --xdp=0.8 --feat_dropout=0.6 --label_dropout=0.2 --use_embedding=True --batch_size=512 --lr=0.0015
```
### C.ele
```
python main.py --dataset=Celegans --xdp=0.8 --feat_dropout=0.05 --label_dropout=0.6 --use_embedding=True --batch_size=512 --weight_decay=0.001 --use_degree=mlp --lr=0.0015
```
### Power
```
python main.py --dataset=Power --xdp=0.8 --feat_dropout=0.05 --label_dropout=0.05 --use_embedding=True --batch_size=512 --weight_decay=0.001 --use_degree=mlp --lr=0.0015
```
### Router
```
python main.py --dataset=Router --xdp=0.8 --feat_dropout=0.6 --label_dropout=0.05 --use_embedding=True --batch_size=512 --weight_decay=0.01 --use_degree=mlp --lr=0.01 --batchnorm_affine=False
```
### E.coli
```
python main.py --dataset=Ecoli --xdp=0.8 --feat_dropout=0.2 --label_dropout=0.6 --use_embedding=True --batch_size=512 --lr=0.0015
```
### CS
```
python main.py --dataset=cs --xdp=0.5 --feat_dropout=0.2 --label_dropout=0.2 --batch_size=2048 --minimum_degree_onehot=60 --lr=0.01 --encoder=puregcn --patience=40
```
### Physics
```
python main.py --dataset=physics --xdp=0.1 --label_dropout=0.6 --batch_size=2048 --use_degree=mlp --minimum_degree_onehot=60 --batchnorm_affine=False --patience=40
```
### Computers
```
python main.py --dataset=computers --xdp=0.1 --feat_dropout=0.05 --label_dropout=0.6 --batch_size=2048 --use_degree=mlp --minimum_degree_onehot=100 --patience=40
```
### Photo
```
python main.py --dataset=photos --xdp=0.1 --feat_dropout=0.05 --label_dropout=0.05 --batch_size=2048 --use_degree=mlp --patience=40
```
### Collab
```
python main.py --dataset=ogbl-collab --batch_size=4096 --use_degree=mlp --patience=40 --log_steps=1 --minimum_degree_onehot=100 --year=2010 --use_valedges_as_input=True --xdp=0.5 --label_dropout=0.05 --lr=0.01 --signature_dim=2048
```
### Collab (no feat)
```
python main.py --dataset=ogbl-collab --use_feature=False --batch_size=8192 --mask_target=True --weight_decay=0 --use_degree=mlp --patience=40 --log_steps=1 --minimum_degree_onehot=50 --year=2010 --use_valedges_as_input=True --signature_dim=6000
```
[/logs/no_feat_ogbl-collab_jobID_769579_PID_606228_1690552809.log/]: #

</details>