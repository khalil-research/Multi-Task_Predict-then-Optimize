#!/usr/bin/env python
# coding: utf-8
"""
Train with different method
"""

import time

import torch
from torch import nn
from tqdm import tqdm

import pyepo
import net
from earlystop import earlyStopper
from gradnorm import gradNorm


def trainSeparated(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with separated model
    """
    m = data_params["node"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    proc = train_params["proc"] # process number for optmodel
    # init
    reg = {}
    loss_log = {}
    elapsed, elapsed_val = 0, 0
    # per task
    for ind, (task, optmodel) in enumerate(optmodels.items()):
        reg[task] = net.reg(p, m)
        # cuda
        if torch.cuda.is_available():
            reg[task] = reg[task].cuda()
        # set optimizer
        optimizer = torch.optim.Adam(reg[task].parameters(), lr=lr)
        # init FY loss
        pfyl = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=5, epsilon=1.0, processes=proc)
        # set stopper
        stopper = earlyStopper(patience=5)
        # train mode
        reg[task].train()
        # init log
        loss_log[task] = []
        # start traning
        tbar = tqdm(range(num_epochs))
        for epoch in tbar:
            tick = time.time()
            # load data
            for data in dataloader:
                x, c, w, z = data
                # cuda
                if torch.cuda.is_available():
                    x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                # forward pass
                cp = reg[task](x)
                loss = pfyl(cp, w[:,ind]).mean()
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # log
                loss_log[task].append(loss.item())
                tbar.set_description("Loss: {:3.4f}".format(loss.item()))
            # time
            tock = time.time()
            elapsed += tock - tick
            # early stop
            if epoch % val_step == 0:
                tick = time.time()
                loss = 0
                with torch.no_grad():
                    for data in dataloader_val:
                        x, c, w, z = data
                        # cuda
                        if torch.cuda.is_available():
                            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                        # forward pass
                        cp = reg[task](x)
                        loss += pfyl(cp, w[:,ind]).mean()
                tock = time.time()
                elapsed_val += tock - tick
                if stopper.stop(loss):
                    break
    return reg, loss_log, elapsed, elapsed_val, None


def trainComb(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with simple combination
    """
    m = data_params["node"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    proc = train_params["proc"] # process number for optmodel
    # init model
    reg = net.reg(p, m)
    mtl = net.mtlPFYL(reg, optmodels.values(), n_samples=5, epsilon=1.0, processes=proc)
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # set stopper
    stopper = earlyStopper(patience=5)
    # train mode
    mtl.train()
    # init log
    loss_log = []
    # start traning
    tbar = tqdm(range(num_epochs))
    elapsed, elapsed_val = 0, 0
    for epoch in tbar:
        tick = time.time()
        # load data
        for data in dataloader:
            # cuda
            if torch.cuda.is_available():
                data = [d.cuda() for d in data]
            # forward pass
            loss = mtl(*data).sum()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_log.append(loss.item())
            tbar.set_description("Loss: {:3.4f}".format(loss.item()))
        # time
        tock = time.time()
        elapsed += tock - tick
        # early stop
        if epoch % val_step == 0:
            tick = time.time()
            loss = 0
            with torch.no_grad():
                for data in dataloader_val:
                    # cuda
                    if torch.cuda.is_available():
                        data = [d.cuda() for d in data]
                    # forward pass
                    loss += mtl(*data).sum()
            tock = time.time()
            elapsed_val += tock - tick
            if stopper.stop(loss):
                break
    return reg, loss_log, elapsed, elapsed_val, None


def trainGradNorm(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with GradNorm
    """
    m = data_params["node"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    lr2 = train_params["lr2"] # learnin rate for weights
    proc = train_params["proc"] # process number for optmodel
    alpha = train_params["alpha"] # hyperparameter of restoring force
    # init model
    reg = net.reg(p, m)
    mtl = net.mtlPFYL(reg, optmodels.values(), n_samples=5, epsilon=1.0, processes=proc)
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train mode
    mtl.train()
    # init log
    loss_log = []
    # start traning
    weights_log, loss_log, elapsed, elapsed_val = gradNorm(net=mtl, layer=reg.linear,
                                                           alpha=alpha, dataloader=dataloader,
                                                           dataloader_val=dataloader_val,
                                                           num_epochs=num_epochs, lr1=lr,
                                                           lr2=5e-3, val_step=val_step)
    return reg, loss_log, elapsed, elapsed_val, weights_log
