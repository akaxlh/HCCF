
# HCCF

This repository contains TensorFlow codes and datasets for the paper:

>Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, Jimmy Xiangji Huang (2020). Hypergraph Contrastive Collaborative Filtering, <a href='https://arxiv.org/abs/2204.12200'>Paper in arXiv</a>. In SIGIR'22, Madrid, Spain, July 11-15, 2022.

## Introduction
Hypergraph Contrastive Collaborative Filtering (HCCF) devises parameterized hypergraph neural network and hypergraph-graph contrastive learning, to relieve the over-smoothing issue for conventional graph neural networks, and address the sparse and skewed data distribution problem in collaborative filtering.

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{hccf2022,
  author    = {Xia, Lianghao and
               Huang, Chao and
	       Xu, Yong and
	       Zhao, Jiashu and
	       Yin, Dawei and
	       Huang, Jimmy Xiangji},
  title     = {Hypergraph Contrastive Collaborative Filtering},
  booktitle = {Proceedings of the 45th International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2022, Madrid,
               Spain, July 11-15, 2022.},
  year      = {2022},
}
```

## Environment
The codes of HCCF are implemented and tested under the following development environment:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

## Datasets
We utilized three datasets to evaluate HCCF: <i>Yelp, MovieLens, </i>and <i>Amazon</i>. Following the common settings of implicit feedback, if user $u_i$ has rated item $v_j$, then the element $(u_i, v_j)$ is set as 1, otherwise 0. We filtered out users and items with too few interactions. The datasets are divided into training set, validation set and testing set by 7:1:2.

## How to Run the Codes
Please unzip the datasets first. Also you need to create the `History/` and the `Models/` directories. The command to train HCCF on the Yelp/MovieLens/Amazon dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.

* Yelp
```
python .\labcode_efficient.py --data yelp --temp 1 --ssl_reg 1e-4
```
* MovieLens
```
python .\labcode_efficient.py --data ml10m --temp 0.1 --ssl_reg 1e-6 --keepRate 1.0 --reg 1e-3
```
* Amazon
```
python .\labcode_efficient.py --data amazon --temp 0.1 --ssl_reg 1e-7 --reg 1e-2
```
Important arguments:
* `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{1e-2, 1e-3, 1e-4, 1e-5}`.
* `ssl_reg`: This is the weight for the hypergraph-graph contrastive learning loss. The value is tuned from `1e-2` to `1e-8`.
* `temp`: This is the temperature factor in the InfoNCE loss in our contrastive learning. The value is selected from `{10, 3, 1, 0.3, 0.1}`.
* `keepRate`: It denotes the rate to keep edges in the graph dropout, which is tuned from `{0.25, 0.5, 0.75, 1.0}`.

## Acknowledgements
This research is supported by the research grants from the Department of Computer Science & Musketeers Foundation Institute of Data Science at the University of Hong Kong, the Natural Sciences & Engineering Research Council (NSERC) of Canada.
