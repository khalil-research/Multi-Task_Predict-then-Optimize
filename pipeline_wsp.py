#!/usr/bin/env python
# coding: utf-8
"""
Experiments pipeline for warcraft
"""

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# random seed
random.seed(135)
np.random.seed(135)
torch.manual_seed(135)

import solver
import data
import train
import plot
import metric

def run(config):
    """
    Train with Warcraft
    """
    # dir
    log_dir, res_dir = getDir(config)
    # build optimization models
    optmodel = getOptModel(k=config.k)
    # generate data
    print("Seting dataloader...")
    # set data loader
    loaders_train, loeaders_val, loaders_test = genDataLoader(config.k, config.data, config.batch)
    # training methods
    if config.algo == "spo":
        print("Using SPO+...")
        train_algos = {
        "mse": train.warcraft.spo.train2S, # 2-stage
        "separated+mse": train.warcraft.spo.trainSeparatedMSE, # separated + MSE
        "comb+mse": train.warcraft.spo.trainCombMSE, # simple combination + MSE
        "gradnorm+mse": train.warcraft.spo.trainGradNormMSE # gradNorm + MSE
        }
    if config.algo == "pfyl":
        print("Using PFYL...")
        train_algos = {
        "separated": train.warcraft.pfyl.trainSeparated, # separated
        "comb": train.warcraft.pfyl.trainComb, # simple combination
        "gradnorm": train.warcraft.pfyl.trainGradNorm # gradNorm
        }
    # init df for total evaluation
    df = []
    # init dict for evaluation per instance
    inst_res = {}
    for modelname, train_func in train_algos.items():
        print("===============================================================")
        print("Start training {}...".format(modelname))
        nnet, loss_log, elapsed, elapsed_val, weights_log = train_func(loaders_train, loeaders_val, optmodel, config)
        print()
        # plot logs
        saveLogFig(modelname, log_dir, loss_log, weights_log, config)
        # eval
        print("Evaluating...")
        total_evals, inst_evals = getEvals(nnet, modelname, loaders_test, optmodel)
        total_evals["Elapsed"], total_evals["Elapsed Val"] = elapsed, elapsed_val
        df.append(total_evals)
        inst_res[modelname] = inst_evals
        print()
    # plot res
    df = pd.DataFrame(df)
    print("Drawing radar plot...")
    saveRadarFig(res_dir, df, config)
    print()
    # save res
    saveRes(res_dir, df, inst_res)
    print("Save results to " + res_dir + "/res.csv")
    print()
    print()


def getDir(config):
    """
    Get dir to save figure and result
    """
    # logs
    log_dir = "./log/warcraft/{}/{}".format(config.algo, config.data)
    os.makedirs(log_dir, exist_ok=True)
    # results
    res_dir = "./res/warcraft/{}/{}".format(config.algo, config.data)
    os.makedirs(res_dir, exist_ok=True)
    return log_dir, res_dir


def getOptModel(k):
    """
    Init optimization model
    """
    grid = (k, k)
    optmodel = solver.warcraftShortestPathModel(grid)
    return optmodel


def genDataLoader(k ,data_size, batch_size):
    """
    Set data loader
    """
    data_dir = "./data/warcraft_shortest_path_oneskin/{}x{}/".format(k,k)
    ## dataset 1: Human
    ind = np.random.randint(10000, size=data_size)
    # tmap
    tmaps_train0 = np.load(data_dir + "/train_maps.npy")[ind]
    tmaps_val0   = np.load(data_dir + "/val_maps.npy")[:1000]
    tmaps_test0  = np.load(data_dir + "/test_maps.npy")
    # cost
    costs_train0 = np.load(data_dir + "/train_vertex_weights.npy")[ind]
    costs_val0   = np.load(data_dir + "/val_vertex_weights.npy")[:1000]
    costs_test0  = np.load(data_dir + "/test_vertex_weights.npy")
    # path
    paths_train0 = np.load(data_dir + "/train_shortest_paths.npy")[ind]
    paths_val0   = np.load(data_dir + "/val_shortest_paths.npy")[:1000]
    paths_test0  = np.load(data_dir + "/test_shortest_paths.npy")
    # dataset
    dataset_train0 = data.mapDataset(tmaps_train0, costs_train0, paths_train0)
    dataset_val0   = data.mapDataset(tmaps_val0, costs_val0, paths_val0)
    dataset_test0  = data.mapDataset(tmaps_test0, costs_test0, paths_test0)
    # dataloader
    loader_train0 = DataLoader(dataset_train0, batch_size=batch_size, shuffle=True)
    loader_val0   = DataLoader(dataset_val0, batch_size=batch_size, shuffle=False)
    loader_test0  = DataLoader(dataset_test0, batch_size=batch_size, shuffle=False)
    ## dataset 2: Naga
    ind = np.random.randint(10000, size=data_size)
    # tmap
    tmaps_train1 = np.load(data_dir + "/train_maps.npy")[ind]
    tmaps_val1   = np.load(data_dir + "/val_maps.npy")[:1000]
    tmaps_test1  = np.load(data_dir + "/test_maps.npy")
    # cost
    costs_train1 = np.load(data_dir + "/train_vertex_weights_1.npy")[ind]
    costs_val1   = np.load(data_dir + "/val_vertex_weights_1.npy")[:1000]
    costs_test1  = np.load(data_dir + "/test_vertex_weights_1.npy")
    # path
    paths_train1 = np.load(data_dir + "/train_shortest_paths_1.npy")[ind]
    paths_val1   = np.load(data_dir + "/val_shortest_paths_1.npy")[:1000]
    paths_test1  = np.load(data_dir + "/test_shortest_paths_1.npy")
    # dataset
    dataset_train1 = data.mapDataset(tmaps_train1, costs_train1, paths_train1)
    dataset_val1   = data.mapDataset(tmaps_val1, costs_val1, paths_val1)
    dataset_test1  = data.mapDataset(tmaps_test1, costs_test1, paths_test1)
    # dataloader
    loader_train1 = DataLoader(dataset_train1, batch_size=batch_size, shuffle=True)
    loader_val1   = DataLoader(dataset_val1, batch_size=batch_size, shuffle=False)
    loader_test1  = DataLoader(dataset_test1, batch_size=batch_size, shuffle=False)
    ## dataset 3: Dwaf
    ind = np.random.randint(10000, size=data_size)
    # tmap
    tmaps_train2 = np.load(data_dir + "/train_maps.npy")[ind]
    tmaps_val2   = np.load(data_dir + "/val_maps.npy")[:1000]
    tmaps_test2  = np.load(data_dir + "/test_maps.npy")
    # cost
    costs_train2 = np.load(data_dir + "/train_vertex_weights_2.npy")[ind]
    costs_val2   = np.load(data_dir + "/val_vertex_weights_2.npy")[:1000]
    costs_test2  = np.load(data_dir + "/test_vertex_weights_2.npy")
    # path
    paths_train2 = np.load(data_dir + "/train_shortest_paths_2.npy")[ind]
    paths_val2   = np.load(data_dir + "/val_shortest_paths_2.npy")[:1000]
    paths_test2  = np.load(data_dir + "/test_shortest_paths_2.npy")
    # dataset
    dataset_train2 = data.mapDataset(tmaps_train2, costs_train2, paths_train2)
    dataset_val2   = data.mapDataset(tmaps_val2, costs_val2, paths_val2)
    dataset_test2  = data.mapDataset(tmaps_test2, costs_test2, paths_test2)
    # dataloader
    loader_train2 = DataLoader(dataset_train2, batch_size=batch_size, shuffle=True)
    loader_val2   = DataLoader(dataset_val2, batch_size=batch_size, shuffle=False)
    loader_test2  = DataLoader(dataset_test2, batch_size=batch_size, shuffle=False)
    # tuple
    loaders_train = (loader_train0, loader_train1, loader_train2)
    loeaders_val  = (loader_val0, loader_val1, loader_val2)
    loaders_test  = (loader_test0, loader_test1, loader_test2)
    return loaders_train, loeaders_val, loaders_test


def saveLogFig(modelname, log_dir, loss_log, weights_log, config):
    """
    Save plot of loss & weights
    """
    # plot loss
    if (modelname == "separated") or (modelname == "separated+mse"):
        for t, task in enumerate(loss_log):
            loss_fig = plot.plotLoss(loss_log[task])
            # save
            loss_fig.savefig(log_dir + "/loss_{}_{}.pdf".format(modelname, t+1), dpi=300)
            loss_fig.savefig(log_dir + "/loss_{}_{}.png".format(modelname, t+1), dpi=300)
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
        labels = []
        for t in range(config.n_tasks):
            labels.append("Regret {}".format(t+1))
            if modelname[-3:] == "mse":
                labels.append("MSE {}".format(t+1))
        weights_fig = plot.plotWeights(weights_log, labels)
        # save
        weights_fig.savefig(log_dir+"/weights_{}.pdf".format(modelname), dpi=300)
        weights_fig.savefig(log_dir+"/weights_{}.png".format(modelname), dpi=300)
        # close
        plt.close(weights_fig)


def getEvals(nnet, modelname, dataloaders, optmodel):
    """
    Get evaluations
    """
    # eval
    res = metric.evalMultiCostModel(nnet, dataloaders, optmodel)
    # init df
    inst_evals = {}
    total_evals = {"Method": modelname}
    for t in range(nnet.n_tasks):
        # init record per task
        df = {}
        # mse
        df["MSE"] = res["MSE"][:,t]
        avg_mse = res["MSE"][:,t].mean()
        med_mse = np.median(res["MSE"][:,t])
        # regret
        df["Regret"] = res["Regret"][:,t]
        avg_regret = res["Regret"][:,t].mean()
        med_regret = np.median(res["Regret"][:,t])
        df["Relative Regret"] = res["Relative Regret"][:,t]
        avg_relregret = res["Relative Regret"][:,t].mean()
        med_relregret = np.median(res["Relative Regret"][:,t])
        # optimality rate
        df["Optimal"] = res["Optimal"][:,t]
        optimal = res["Optimal"][:,t].mean()
        print("Task {}: MSE: {:.2f}, Avg Regret: {:.4f}, Avg Rel Regret: {:.2f}%, Optimality Rate: {:.2f}%".\
               format(t, avg_mse, avg_regret, avg_relregret*100, optimal*100))
        total_evals["Task {} MSE".format(t+1)] = avg_mse
        total_evals["Task {} Med MSE".format(t+1)] = med_mse
        total_evals["Task {} Avg Regret".format(t+1)] = avg_regret
        total_evals["Task {} Med Regret".format(t+1)] = med_regret
        total_evals["Task {} Avg Relative Regret".format(t+1)] = avg_relregret
        total_evals["Task {} Med Relative Regret".format(t+1)] = med_relregret
        total_evals["Task {} Optimality Rate".format(t+1)] = optimal
        # record
        inst_evals["Task {}".format(t+1)] = pd.DataFrame(df)
    return total_evals, inst_evals


def saveRadarFig(res_dir, df, config):
    """
    Draw and save radarplot for result comparison
    """
    # draw
    fig = plot.plotMultiCostPerfRadar(df, config)
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
    parser.add_argument("--data",
                        type=int,
                        default=500,
                        help="training data size")
    parser.add_argument("--k",
                        type=int,
                        default=12,
                        help="image size")
    # training configuration
    parser.add_argument("--algo",
                        type=str,
                        choices=["spo", "pfyl"],
                        default="spo",
                        help="training algorithm")
    parser.add_argument("--batch",
                        type=int,
                        default=70,
                        help="batch size")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-5,
                        help="learning rate")
    parser.add_argument("--iters",
                        type=int,
                        default=int(5e5),
                        help="max iterations")
    parser.add_argument("--proc",
                        type=int,
                        default=1,
                        help="number of processor for optimization")

    # get configuration
    config = parser.parse_args()
    config.n_tasks = 3
    config.epoch = config.iters//config.data # max epoch
    config.vstep = max(1, config.epoch//50) # validation step

    # run
    run(config)
