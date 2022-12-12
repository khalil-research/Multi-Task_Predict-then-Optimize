#!/usr/bin/env python
# coding: utf-8
"""
Experiments pipeline
"""

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy import stats
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# random seed
random.seed(135)
np.random.seed(135)
torch.manual_seed(135)

import pyepo
import graph
import solver
import data
import train
import metric
import plot

def run(data_params, task_params, train_params):
    """
    Train with multiple tasks
    """
    # dir
    log_dir, res_dir = getDir(data_params, task_params, train_params)
    # generate data
    print("Generating data...")
    x, c = genData(data_params)
    # build optimization models
    optmodels = getOptModel(data_params, task_params)
    # set data loader
    loader_train, loader_val, loader_test = genDataLoader(x, c, optmodels, train_params)
    print()
    # training methods
    if train_params["algo"] == "spo":
        print("Using SPO+...")
        train_algos = {
        "mse": train.graph.spo.train2S, # 2-stage
        "separated+mse": train.graph.spo.trainSeparatedMSE, # separated with mse
        "comb+mse": train.graph.spo.trainCombMSE, # simple combination with mse
        "gradnorm+mse": train.graph.spo.trainGradNormMSE, # GradNorm with mse
        }
    if  train_params["algo"] == "pfyl":
        print("Using PFYL...")
        train_algos = {
        "separated": train.graph.pfyl.trainSeparated, # separated
        "comb": train.graph.pfyl.trainComb, # simple combination
        "gradnorm": train.graph.pfyl.trainGradNorm, # GradNorm
        }
    # init df for total evaluation
    df = []
    # init dict for evaluation per instance
    inst_res = {}
    # train
    for modelname, train_func in train_algos.items():
        print("===============================================================")
        print("Start training {}...".format(modelname))
        reg, loss_log, elapsed, elapsed_val, weights_log = train_func(loader_train, loader_val, optmodels, data_params, train_params)
        print()
        # plot logs
        saveLogFig(modelname, log_dir, loss_log, weights_log, optmodels)
        # eval
        print("Evaluating...")
        total_evals, inst_evals = getEvals(reg, modelname, loader_test, optmodels)
        total_evals["Elapsed"] = elapsed
        total_evals["Elapsed Val"] = elapsed_val
        df.append(total_evals)
        inst_res[modelname] = inst_evals
        print()
    df = pd.DataFrame(df)
    # plot res
    print("Drawing radar plot...")
    saveRadarFig(res_dir, df, optmodels)
    print()
    # save res
    saveRes(res_dir, df, inst_res)
    print()
    print()


def getDir(data_params, task_params, train_params):
    """
    Get dir to save figure and result
    """
    n = data_params["data"] # number of data
    p = data_params["feat"] # size of feature
    deg = data_params["deg"] # polynomial degree
    n_sp = task_params["n_sp"] # number of sp tasks
    n_tsp = task_params["n_tsp"] # number of tsp tasks
    algo = train_params["algo"] # training algo
    # logs
    log_dir = "./log/{}sp{}tsp/{}/n{}p{}deg{}".format(n_sp, n_tsp, algo, n, p, deg)
    os.makedirs(log_dir, exist_ok=True)
    # results
    res_dir = "./res/{}sp{}tsp/{}/n{}p{}deg{}".format(n_sp, n_tsp, algo, n, p, deg)
    os.makedirs(res_dir, exist_ok=True)
    return log_dir, res_dir


def genData(data_params):
    """
    Generate data (features and costs)
    """
    m = data_params["node"] # number of nodes
    n = data_params["data"] # number of data
    p = data_params["feat"] # size of feature
    deg = data_params["deg"] # polynomial degree
    e = data_params["noise"] # noise half-width
    # features and costs
    x, c = pyepo.data.tsp.genData(n+100+1000, p, m, deg=deg, noise_width=e)
    return x, c


def getOptModel(data_params, task_params):
    """
    Build optimization model for multiple task
    """
    m = data_params["node"] # number of nodes
    n_sp = task_params["n_sp"] # number of sp tasks
    n_tsp = task_params["n_tsp"] # number of tsp tasks
    # init graph
    g = nx.complete_graph(m)
    # init optmodel dic
    optmodels = {}
    # shortest path
    spgraph_gen = graph.genSubgraphSP(g, len(g.edges)//8)
    for i in range(n_sp):
        # subgraph
        _, edges, s, t = next(spgraph_gen)
        # model
        optmodel = solver.shortestPathModelLP(g, edges=edges, source=s, target=t)
        optmodels["SP {}".format(i+1)] = optmodel
    # tsp
    tspgraph_gen = graph.genSubgraphTSP(g)
    for i in range(n_tsp):
        # subgraph
        _, nodes = next(tspgraph_gen)
        # model
        optmodel = solver.tspModel(g, nodes)
        optmodels["TSP {}".format(i+1)] = optmodel
    return optmodels


def genDataLoader(x, c, optmodels, train_params):
    """
    Set data loader with solving optimal solutions
    """
    # data split
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=1000, random_state=135)
    x_train, x_val, c_train, c_val = train_test_split(x_train, c_train, test_size=100, random_state=246)
    # dataset
    dataset_train, dataset_val, dataset_test = data.buildDataset(x_train, x_val,
                                                                 x_test, c_train,
                                                                 c_val, c_test,
                                                                 optmodels.values())
    # get data loader
    batch_size = train_params["batch"]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_val, loader_test


def saveLogFig(modelname, log_dir, loss_log, weights_log, optmodels):
    """
    Save plot of loss & weights
    """
    # plot loss
    if (modelname == "separated") or (modelname == "separated+mse"):
        for task in loss_log:
            loss_fig = plot.plotLoss(loss_log[task])
            # save
            task = task.lower().replace(" ", "")
            loss_fig.savefig(log_dir + "/loss_{}_{}.pdf".format(modelname, task), dpi=300)
            loss_fig.savefig(log_dir + "/loss_{}_{}.png".format(modelname, task), dpi=300)
            # close
            plt.close(loss_fig)
    else:
        loss_fig = plot.plotLoss(loss_log)
        # save
        loss_fig.savefig(log_dir + "/loss_{}.pdf".format(modelname), dpi=300)
        loss_fig.savefig(log_dir + "/loss_{}.png".format(modelname), dpi=300)
        # close
        plt.close(loss_fig)
    # plot adaptive weights
    if weights_log is not None:
        labels = [l for l in optmodels.keys()] + ["MSE"]  # labels
        weights_fig = plot.plotWeights(weights_log, labels)
        # save
        weights_fig.savefig(log_dir+"/weights_{}.pdf".format(modelname), dpi=300)
        weights_fig.savefig(log_dir+"/weights_{}.png".format(modelname), dpi=300)
        # close
        plt.close(weights_fig)


def getEvals(reg, modelname, dataloader, optmodels):
    """
    Get evaluations
    """
    inst_evals = {}
    if (modelname == "separated") or (modelname == "separated+mse"):
        # init row
        total_evals = {"Method": modelname, "MSE":[], "Med MSE":[]}
        for i, task in enumerate(optmodels):
            # init record per task
            df = {}
            # eval
            res = metric.evalModel(reg[task], dataloader, optmodels.values())
            # mse
            df["MSE"] = res["MSE"]
            mse = res["MSE"].mean()
            med_mse = np.median(res["MSE"])
            total_evals["MSE"].append(mse)
            total_evals["Med MSE"].append(med_mse)
            # regret
            df["Regret"] = res["Regret"][:,i]
            avg_regret = res["Regret"][:,i].mean()
            med_regret = np.median(res["Regret"][:,i])
            df["Relative Regret"] = res["Relative Regret"][:,i]
            avg_relregret = res["Relative Regret"][:,i].mean()
            med_relregret = np.median(res["Relative Regret"][:,i])
            # optimality rate
            df["Optimal"] = res["Optimal"][:,i]
            optimal = res["Optimal"][:,i].mean()
            print("Task {}: MSE: {:.2f}, Avg Regret: {:.4f}, Avg Rel Regret: {:.2f}%, Optimality Rate: {:.2f}%".\
                  format(task, mse, avg_regret, avg_relregret*100, optimal*100))
            total_evals["{} Avg Regret".format(task)] = avg_regret
            total_evals["{} Med Regret".format(task)] = med_regret
            total_evals["{} Avg Relative Regret".format(task)] = med_relregret
            total_evals["{} Med Relative Regret".format(task)] = avg_relregret
            total_evals["{} Optimality Rate".format(task)] = optimal
            # record
            inst_evals[task] = pd.DataFrame(df)
    else:
        # eval
        res = metric.evalModel(reg, dataloader, optmodels.values())
        # init row
        total_evals = {
        "Method": modelname,
        "MSE": res["MSE"].mean(),
        "Med MSE": np.median(res["MSE"])
        }
        print("Mean Squared Error: {:.4f}".format(total_evals["MSE"]))
        for i, task in enumerate(optmodels):
            # init record per task
            df = {}
            # mse
            df["MSE"] = res["MSE"]
            # regret
            df["Regret"] = res["Regret"][:,i]
            avg_regret = res["Regret"][:,i].mean()
            med_regret = np.median(res["Regret"][:,i])
            df["Relative Regret"] = res["Relative Regret"][:,i]
            avg_relregret = res["Relative Regret"][:,i].mean()
            med_relregret = np.median(res["Relative Regret"][:,i])
            # optimality rate
            df["Optimal"] = res["Optimal"][:,i]
            optimal = res["Optimal"][:,i].mean()
            print("Task {}: Avg Regret: {:.4f}, Avg Rel Regret: {:.2f}%, Optimality Rate: {:.2f}%".\
                  format(task, avg_regret, avg_relregret*100, optimal*100))
            total_evals["{} Avg Regret".format(task)] = avg_regret
            total_evals["{} Med Regret".format(task)] = med_regret
            total_evals["{} Avg Relative Regret".format(task)] = avg_relregret
            total_evals["{} Med Relative Regret".format(task)] = avg_relregret
            total_evals["{} Optimality Rate".format(task)] = optimal
            # record
            inst_evals[task] = pd.DataFrame(df)
    return total_evals, inst_evals


def saveRadarFig(res_dir, df, optmodels):
    """
    Draw and save radarplot for result comparison
    """
    # draw average performence
    fig = plot.plotPerfRadar(df, optmodels)
    # save
    dir = res_dir+"/radar.png"
    fig.savefig(dir, dpi=300)
    print("Save radar plot for performence to " + dir)
    dir = res_dir+"/radar.pdf"
    fig.savefig(dir, dpi=300)
    print("Save radar plot for performence to " + dir)
    # close
    plt.close(fig)


def saveRes(res_dir, df, inst_res):
    """
    Save results
    """
    df.to_csv(res_dir + "/res.csv")
    print("Save results to " + res_dir + "/res.csv")
    for method in inst_res:
        for task in inst_res[method]:
            res_path = res_dir + "/res_{}_{}.csv".format(method, task.lower().replace(" ", ""))
            inst_res[method][task].to_csv(res_path)
            print("Save results to " + res_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # data configuration
    parser.add_argument("--node",
                        type=int,
                        default=30,
                        help="number of nodes in graph")
    parser.add_argument("--data",
                        type=int,
                        default=100,
                        help="training data size")
    parser.add_argument("--feat",
                        type=int,
                        default=10,
                        help="feature size")
    parser.add_argument("--deg",
                        type=int,
                        default=4,
                        help="features polynomial degree")
    parser.add_argument("--noise",
                        type=float,
                        default=0.5,
                        help="noise half-width")

    # task configuration
    parser.add_argument("--n_sp",
                        type=int,
                        default=3,
                        help="number of the shortest path tasks")
    parser.add_argument("--n_tsp",
                        type=int,
                        default=1,
                        help="number of the TSP tasks")

    # training configuration
    parser.add_argument("--algo",
                        type=str,
                        choices=["spo", "pfyl"],
                        default="spo",
                        help="training algorithm")
    parser.add_argument("--iter",
                        type=int,
                        default=30000,
                        help="max iterations")
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-1,
                        help="learning rate")
    parser.add_argument("--lr2",
                        type=float,
                        default=5e-3,
                        help="learning rate for GradNorm weights")
    parser.add_argument("--alpha",
                        type=float,
                        default=1e-1,
                        help="GradNorm hyperparameter of restoring force")
    parser.add_argument("--proc",
                        type=int,
                        default=1,
                        help="number of processor for optimization")

    # get configuration
    config = parser.parse_args()
    # get args
    data_args = {
    "node":  config.node,
    "data":  config.data,
    "feat":  config.feat,
    "deg":   config.deg,
    "noise": config.noise,
    }
    task_args = {
    "n_sp":  config.n_sp,
    "n_tsp": config.n_tsp,
    }
    train_args = {
    "algo":  config.algo,
    "batch": config.batch,
    "epoch": config.iter//config.data, # 30000 max iters
    "lr":    config.lr,
    "lr2":   config.lr2,
    "alpha": config.alpha,
    "proc":  config.proc,
    "val_step": max(config.iter//config.data//60, 1) # validation step
    }

    # run
    run(data_args, task_args, train_args)

# python pipeline.py --algo pfyl
