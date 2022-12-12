#!/usr/bin/env python
# coding: utf-8

"""
Evaluation
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from pyepo import EPO

def evalModel(reg, dataloader, solvers):
    """
    Eval with a neural network
    """
    costs_pred = []
    # eval mode
    reg.eval()
    # load data
    for data in dataloader:
        x, c, w, z = data
        # cuda
        if next(reg.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
        # predict
        cp = reg(x).to("cpu").detach().numpy()
        costs_pred.append(cp)
    costs_pred = np.vstack(costs_pred)
    # eval
    res = evaluate(costs_pred, dataloader.dataset, solvers)
    return res

def evalMultiCostModel(nnet, dataloaders, solver):
    """
    Eval with a neural network with mutiple cost
    """
    # init res
    res = {"MSE":[], "Regret":[], "Relative Regret":[], "Optimal":[]}
    # eval mode
    nnet.eval()
    for i in range(nnet.n_tasks):
        costs_pred = []
        # load data
        for x, c, w, z in dataloaders[i]:
            # cuda
            if next(nnet.parameters()).is_cuda:
                x = x.cuda()
            # predict
            cp = nnet(x).to("cpu").detach().numpy()
            costs_pred.append(cp[i])
        costs_pred = np.vstack(costs_pred)
        # eval
        cur_res = evaluate(costs_pred, dataloaders[i].dataset, [solver])
        for key in res:
            res[key].append(cur_res[key].reshape(-1,1))
    # format
    for key in res:
        res[key] = np.hstack(res[key])
    return res


def evaluate(costs_pred, dataset, solvers):
    """
    Eval with given prediction
    """
    # init res
    res = {"MSE":[], "Regret":[], "Relative Regret":[], "Optimal":[]}
    # data
    costs = dataset.costs
    objs = dataset.objs
    for i in tqdm(range(len(costs))):
        cp = costs_pred[i].reshape(-1)
        c = costs[i].reshape(-1)
        z = objs[i].reshape(-1)
        # mse
        res["MSE"].append(np.mean((c - cp) ** 2))
        # init metrics
        regret = np.zeros(len(solvers))
        rel_regret = np.zeros(len(solvers))
        optimal = np.zeros(len(solvers))
        for j, optmodel in enumerate(solvers):
            # set obj
            optmodel.setObj(cp)
            # solve
            wp, _ = optmodel.solve()
            # obj with true cost
            zp = np.dot(wp, c)
            # regret
            if optmodel.modelSense == EPO.MINIMIZE:
                r = zp - z[j]
            if optmodel.modelSense == EPO.MAXIMIZE:
                r = z[j] - zp
            regret[j] = r
            # rel regret
            rel_regret[j] = r / (abs(z[j]) + 1e-7)
            # optimality
            optimal[j] = abs(r) < 1e-5
        res["Regret"].append(regret)
        res["Relative Regret"].append(rel_regret)
        res["Optimal"].append(optimal)
    # numpy
    res["MSE"] = np.array(res["MSE"])
    res["Regret"] = np.array(res["Regret"])
    res["Relative Regret"] = np.array(res["Relative Regret"])
    res["Optimal"] = np.array(res["Optimal"])
    return res
