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


def trainSeparated(dataloaders, dataloaders_val, optmodel, config):
    """
    Train with separated model
    """
    # init net
    nnet = net.partialResNetSeperated(config.k, config.n_tasks)
    mtl = net.partialResNetPFYL(nnet, optmodel, n_samples=1, epsilon=1.0, processes=config.proc)
    # cuda
    if torch.cuda.is_available():
        nnet = nnet.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(nnet.parameters(), lr=config.lr)
    # set stopper
    stopper = earlyStopper(patience=2)
    # init log
    loss_log = {"Task {}".format(t+1):[] for t in range(config.n_tasks)}
    # init elpased time
    elapsed, elapsed_val = 0, 0
    # epoch
    tbar = tqdm(range(config.epoch))
    for epoch in tbar:
        nnet.train()
        tick = time.time()
        for data in zip(*dataloaders):
            # unzip data
            xs = torch.stack([d[0] for d in data]) # feat
            cs = torch.stack([d[1] for d in data]) # cost
            ws = torch.stack([d[2] for d in data]) # sol
            zs = torch.stack([d[3] for d in data]) # obj
            # cuda
            if torch.cuda.is_available():
                xs, ws = xs.cuda(), ws.cuda()
            loss = mtl(xs, cs, ws, zs)
            for t in range(config.n_tasks):
                loss_log["Task {}".format(t+1)].append((loss[t]).item())
            loss = loss.sum()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            tbar.set_description("Loss: {:3.4f}".format(loss.item()))
        # time
        tock = time.time()
        elapsed += tock - tick
        # learning rate decay
        if (epoch == int(config.epoch*0.6)) or (epoch == int(config.epoch*0.8)):
            for g in optimizer.param_groups:
                g['lr'] /= 10
        # early stop
        #nnet.eval()
        #if epoch % config.vstep == 0:
        #    tick = time.time()
        #    loss = 0
        #    with torch.no_grad():
        #        for data in zip(*dataloaders):
        #            # unzip data
        #            xs = torch.stack([d[0] for d in data]) # feat
        #            cs = torch.stack([d[1] for d in data]) # cost
        #            ws = torch.stack([d[2] for d in data]) # sol
        #            zs = torch.stack([d[3] for d in data]) # obj
        #            # cuda
        #            if torch.cuda.is_available():
        #                xs, cs, ws, zs = xs.cuda(), cs.cuda(), ws.cuda(), zs.cuda()
        #            # forward pass
        #            loss += mtl(xs, cs, ws, zs).sum().item()
        #    tock = time.time()
        #    elapsed_val += tock - tick
        #    if stopper.stop(loss):
        #        break
    return nnet, loss_log, elapsed, elapsed_val, None


def trainComb(dataloaders, dataloaders_val, optmodel, config):
    """
    Train with simple combination
    """
    # init net
    nnet = net.partialResNetMTL(config.k, config.n_tasks)
    mtl = net.partialResNetPFYL(nnet, optmodel, n_samples=1, epsilon=1.0, processes=config.proc)
    # cuda
    if torch.cuda.is_available():
        nnet = nnet.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(nnet.parameters(), lr=config.lr)
    # set stopper
    stopper = earlyStopper(patience=2)
    # init log
    loss_log = []
    # init elpased time
    elapsed, elapsed_val = 0, 0
    # epoch
    tbar = tqdm(range(config.epoch))
    for epoch in tbar:
        nnet.train()
        tick = time.time()
        for data in zip(*dataloaders):
            # unzip data
            xs = torch.stack([d[0] for d in data]) # feat
            cs = torch.stack([d[1] for d in data]) # cost
            ws = torch.stack([d[2] for d in data]) # sol
            zs = torch.stack([d[3] for d in data]) # obj
            # cuda
            if torch.cuda.is_available():
                xs, ws = xs.cuda(), ws.cuda()
            # forward
            loss = mtl(xs, cs, ws, zs).sum()
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
        # learning rate decay
        if (epoch == int(config.epoch*0.6)) or (epoch == int(config.epoch*0.8)):
            for g in optimizer.param_groups:
                g['lr'] /= 10
        # early stop
        #nnet.eval()
        #if epoch % config.vstep == 0:
        #    tick = time.time()
        #    loss = 0
        #    with torch.no_grad():
        #        for data in zip(*dataloaders):
        #            # unzip data
        #            xs = torch.stack([d[0] for d in data]) # feat
        #            cs = torch.stack([d[1] for d in data]) # cost
        #            ws = torch.stack([d[2] for d in data]) # sol
        #            zs = torch.stack([d[3] for d in data]) # obj
        #            # cuda
        #            if torch.cuda.is_available():
        #                xs, cs, ws, zs = xs.cuda(), cs.cuda(), ws.cuda(), zs.cuda()
        #            # forward pass
        #            loss += mtl(xs, cs, ws, zs).sum().item()
        #    tock = time.time()
        #    elapsed_val += tock - tick
        #    if stopper.stop(loss):
        #        break
    return nnet, loss_log, elapsed, elapsed_val, None


def trainGradNorm(dataloaders, dataloaders_val, optmodel, config):
    """
    Train with GradNorm with MSE
    """
    # init net
    nnet = net.partialResNetMTL(config.k, config.n_tasks)
    mtl = net.partialResNetPFYL(nnet, optmodel, n_samples=1, epsilon=1.0, processes=config.proc)
    # cuda
    if torch.cuda.is_available():
        nnet = nnet.cuda()
    # start traning
    weights_log, loss_log, elapsed, elapsed_val = gradNorm(net=mtl, layer=nnet.block,
                                                           alpha=0.1, dataloader=dataloaders,
                                                           dataloader_val=dataloaders_val,
                                                           num_epochs=config.epoch, lr1=config.lr,
                                                           lr2=5e-3, val_step=config.vstep,
                                                           earlystop=False, lr_decay=True)
    return nnet, loss_log, elapsed, elapsed_val, weights_log
