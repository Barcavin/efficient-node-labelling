

## Reproduce

## USAir
```
python main.py --dataset=USAir --label_dropout=0.2 --use_feature=False --use_embedding=True --batch_size=1024 --predictor=DP+combine --use_degree=mlp --minimum_degree_onehot=100
```
## NS
```
python main.py --dataset=NS --label_dropout=0.2 --use_feature=False --batch_size=512 --predictor=DP+combine --use_degree=mlp --lr=0.01 --batchnorm_affine=False
```
## PB
```
python main.py --dataset=PB --label_dropout=0.2 --use_feature=False --batch_size=1024 --predictor=DP+combine --use_degree=mlp --minimum_degree_onehot=50 --lr=0.01
```
## Yeast
```
python main.py --dataset=Yeast --label_dropout=0.2 --use_feature=False --batch_size=512 --predictor=DP+combine --use_degree=RA --lr=0.01
```
## C.ele
```
python main.py --dataset=Celegans --use_feature=False --use_embedding=True --batch_size=1024 --predictor=DP+combine --weight_decay=0.001 --use_degree=mlp --minimum_degree_onehot=100 --lr=0.01
```
## Power
```
python main.py --dataset=Power --use_feature=False --batch_size=512 --predictor=DP+combine --weight_decay=0.001 --use_degree=mlp --minimum_degree_onehot=50 --lr=0.01 --batchnorm_affine=False
```
## Router
```
python main.py --dataset=Router --label_dropout=0.2 --use_feature=False --use_embedding=True --batch_size=512 --predictor=DP+combine --weight_decay=0.01 --use_degree=mlp --lr=0.01
```
## E.coli
```
python main.py --dataset=Ecoli --label_dropout=0.6 --use_feature=False --use_embedding=True --batch_size=512 --predictor=DP+combine --use_degree=RA --minimum_degree_onehot=50
```
## CS
```
python main.py --dataset=cs --xdp=0.5 --feat_dropout=0.2 --label_dropout=0.2 --batch_size=2048 --predictor=DP+combine --minimum_degree_onehot=60 --lr=0.01 --encoder=puregcn --patience=40
```
## Physics
```
python main.py --dataset=physics --xdp=0.1 --label_dropout=0.6 --batch_size=2048 --predictor=DP+combine --use_degree=mlp --minimum_degree_onehot=60 --batchnorm_affine=False --patience=40
```
## Computers
```
python main.py --dataset=computers --xdp=0.1 --feat_dropout=0.05 --label_dropout=0.6 --batch_size=2048 --predictor=DP+combine --use_degree=mlp --minimum_degree_onehot=100 --patience=40
```
## Photo
```
python main.py --dataset=photos --xdp=0.1 --feat_dropout=0.05 --label_dropout=0.05 --batch_size=2048 --predictor=DP+combine --use_degree=mlp --patience=40