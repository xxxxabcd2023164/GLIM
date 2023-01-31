# GLIM 

This repo is for source code of paper "Liberate Pseudo Labels from Over-Dependence: Label Information Migration on Sparsely Labeled Graphs".

## Environment

- python == 3.7.10
- pytorch == 1.8.1
- networkx == 2.6.3
- numpy == 1.19.2
- torch_geometric == 2.0.3
- pandas == 1.2.2

## Main Methods

Here we provide an implementation of GLIM in PyTorch, along with an execution example on Cora datasets.  Our code on all datasets will be released for further study after the paper is accepted.
```python
python train_model.py --dataset=Cora --sparse_threshold=0.05 --dense_threshold=0.9 --pseudo_rate=0.6
```




