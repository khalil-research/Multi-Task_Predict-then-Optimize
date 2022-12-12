#!/usr/bin/env python
# coding: utf-8
"""
Combined dataset
"""

import numpy as np
from scipy.spatial import distance
import torch
from torch.utils.data import Dataset

from pyepo.data.dataset import optDataset

def buildDataset(x_train, x_val, x_test, c_train, c_val, c_test, solvers):
    # get datasets
    datasets_train, datasets_val, datasets_test = [], [], []
    for solver in solvers:
        datasets_train.append(optDataset(solver, x_train, c_train))
        datasets_val.append(optDataset(solver, x_val, c_val))
        datasets_test.append(optDataset(solver, x_test, c_test))
    # combined data
    # train data
    dataset_train_comb = datasets_train[0]
    # combine sols & obj
    sols, objs = [], []
    for dataset in datasets_train:
        sols.append(dataset.sols)
        objs.append(dataset.objs)
    sols = np.stack(sols, axis=1)
    objs = np.stack(objs, axis=1)
    dataset_train_comb.sols = sols
    dataset_train_comb.objs = objs
    # val data
    dataset_val_comb = datasets_val[0]
    # combine sols & obj
    sols, objs = [], []
    for dataset in datasets_val:
        sols.append(dataset.sols)
        objs.append(dataset.objs)
    sols = np.stack(sols, axis=1)
    objs = np.stack(objs, axis=1)
    dataset_val_comb.sols = sols
    dataset_val_comb.objs = objs
    # test data
    dataset_test_comb = datasets_test[0]
    # combine sols & obj
    sols, objs = [], []
    for dataset in datasets_test:
        sols.append(dataset.sols)
        objs.append(dataset.objs)
    sols = np.stack(sols, axis=1)
    objs = np.stack(objs, axis=1)
    dataset_test_comb.sols = sols
    dataset_test_comb.objs = objs
    return dataset_train_comb, dataset_val_comb, dataset_test_comb


class mapDataset(Dataset):
    def __init__(self, tmaps, costs, paths):
        self.tmaps = tmaps
        self.costs = costs
        self.paths = paths
        self.objs = (costs * paths).sum(axis=(1,2)).reshape(-1,1)

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, ind):
        return (
            torch.FloatTensor(self.tmaps[ind].transpose(2, 0, 1)/255).detach(), # image
            torch.FloatTensor(self.costs[ind]).reshape(-1),
            torch.FloatTensor(self.paths[ind]).reshape(-1),
            torch.FloatTensor(self.objs[ind]),
        )
