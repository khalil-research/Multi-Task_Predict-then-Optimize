#!/usr/bin/env python
# coding: utf-8
"""
Visualization
"""

from matplotlib import pyplot as plt
import scienceplots
import tol_colors as tc
import numpy as np
plt.style.reload_library()
plt.style.use("science")

def plotLoss(loss_log):
    """
    Plot learning curve
    """
    # draw plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(loss_log, color="c", lw=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Iters", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title("Learning Curve", fontsize=18)
    return fig


def plotWeights(log_weights, labels):
    """
    Plot weights during training
    """
    cmap = tc.tol_cmap("rainbow_PuRd")(np.linspace(0, 1, len(labels)+2))[1:-1]
    fig = plt.figure(figsize=(12, 8))
    for i in range(log_weights.shape[1]):
        plt.plot(log_weights[:,i], lw=2, label=labels[i], color=cmap[i])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, len(log_weights)+len(log_weights)//20)
    plt.xlabel("Iters", fontsize=18)
    plt.ylabel("Weights", fontsize=18)
    plt.title("Adaptive Weights During Training for Each Task", fontsize=18)
    plt.legend(fontsize=12)
    return fig


def plotPerfRadar(df, optmodels):
    """
    Draw radar plot for performence
    """
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(polar=True)
    # data
    df = df.copy()
    for i in df.index:
        if type(df.at[i,"MSE"]) is list:
            df.at[i,"MSE"] = np.mean(df.at[i,"MSE"])
    # tol color
    colors = {"mse": "#332288", "separated":"#88ccee", "separated+mse":"#44aa99",
              "comb": "#117733", "comb+mse": "#999933", "gradnorm": "#ddcc77",
              "gradnorm+mse":"#cc6677"}
    # categories
    categories = list(optmodels.keys()) + ["MSE"]
    categories.append(categories[0])
    # label location
    label_loc = np.linspace(start=0, stop=2*np.pi, num=len(categories))
    # plot per method
    for i in df.index:
        mthd = df.at[i,"Method"]
        values = []
        # regret
        for task in optmodels:
            colname = "{} Avg Regret".format(task)
            regret = df.at[i,colname] / np.ceil(df[colname].abs().max())
            values.append(regret)
        # mse
        mse = df.at[i,"MSE"] / np.ceil(df["MSE"].abs().max())
        values.append(mse)
        # plot
        plt.plot(label_loc, values+[values[0]], label=mthd, lw=2, color=colors[mthd])
        plt.scatter(label_loc, values+[values[0]], c=colors[mthd], s=12)
    # labels
    thetas, ys = np.degrees(label_loc), [0.2, 0.4, 0.6, 0.8, 1]
    plt.thetagrids(thetas, labels=categories, fontsize=18)
    plt.yticks(ticks=ys, labels=[""]*5)
    # annotate regret
    for i, task in enumerate(optmodels):
        colname = "{} Avg Regret".format(task)
        max_val = np.ceil(df[colname].abs().max() + 1e-7)
        theta = label_loc[i]
        for y in ys[:-1]:
            text = "{:.1f}".format(max_val * y)
            plt.text(theta, y, text, fontsize=12)
    # annotate mse
    i += 1
    max_val = np.ceil(df["MSE"].abs().max() + 1e-7)
    theta = label_loc[i]
    for y in ys[:-1]:
        text = "{:.1f}".format(max_val * y)
        plt.text(theta, y, text, fontsize=12)
    plt.legend(fontsize=18, bbox_to_anchor=(1, 0.5))
    return fig


def plotMultiCostPerfRadar(df, config):
    """
    Draw radar plot for performence for warcraft multiple costs
    """
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(polar=True)
    # tol color
    colors = {"mse": "#332288", "separated":"#88ccee", "separated+mse":"#44aa99",
              "comb": "#117733", "comb+mse": "#999933", "gradnorm": "#ddcc77",
              "gradnorm+mse":"#cc6677"}
    # categories
    categories = []
    for t in range(config.n_tasks):
        categories.append("Regret {}".format(t+1))
    for t in range(config.n_tasks):
        categories.append("MSE {}".format(t+1))
    categories.append(categories[0])
    # label location
    label_loc = np.linspace(start=0, stop=2*np.pi, num=len(categories))
    for i in df.index:
        mthd = df.at[i,"Method"]
        values = []
        for t in range(config.n_tasks):
            colname = "Task {} Avg Regret".format(t+1)
            regret = df.at[i,colname] / np.ceil(df[colname].abs().max() + 1e-7)
            values.append(regret)
        for t in range(config.n_tasks):
            colname = "Task {} MSE".format(t+1)
            mse = df.at[i,colname] / np.ceil(df[colname].abs().max() + 1e-7)
            values.append(mse)
        # plot
        plt.plot(label_loc, values+[values[0]], label=mthd, lw=2, color=colors[mthd])
        plt.scatter(label_loc, values+[values[0]], c=colors[mthd], s=12)
    # labels
    thetas, ys = np.degrees(label_loc), [0.2, 0.4, 0.6, 0.8, 1]
    plt.thetagrids(thetas, labels=categories, fontsize=18)
    plt.yticks(ticks=ys, labels=[""]*5)
    # annotate regret
    i = 0
    for t in range(config.n_tasks):
        colname = "Task {} Avg Regret".format(t+1)
        max_val = np.ceil(df[colname].abs().max() + 1e-7)
        theta = label_loc[i]
        i += 1
        for y in ys[:-1]:
            text = "{:.1f}".format(max_val * y)
            plt.text(theta, y-0.02, text, fontsize=12)
    # annotate mse
    for t in range(config.n_tasks):
        colname = "Task {} MSE".format(t+1)
        max_val = np.ceil(df[colname].abs().max() + 1e-7)
        theta = label_loc[i]
        i += 1
        for y in ys[:-1]:
            text = "{:.1f}".format(max_val * y)
            plt.text(theta, y-0.02, text, fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=(1, 0.5))
    return fig
