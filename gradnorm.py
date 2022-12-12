#!/usr/bin/env python
# coding: utf-8
"""
Training with GradNorm Algorithm
"""

import time

import numpy as np
import torch
from tqdm import tqdm

from earlystop import earlyStopper

def gradNorm(net, layer, alpha, dataloader, dataloader_val, num_epochs, lr1, lr2, val_step, earlystop=True, lr_decay=False):
    """
    Args:
        net (nn.Module): a multitask network with task loss
        layer (nn.Module): a layers of the full network where appling GradNorm on the weights
        alpha (float): hyperparameter of restoring force
        dataloader (tuple(DataLoader) / DataLoader): training dataloader
        dataloader_val (tuple(DataLoader) / DataLoader): validation dataloader
        num_epochs (int): number of epochs
        lr1（float): learning rate of multitask loss
        lr2（float): learning rate of weights
        val_step (int): validation steps
        lr_decay (bool): learning rate decay flag
    """
    # init log
    weights_log = []
    loss_log = []
    # set optimizer
    optimizer1 = torch.optim.Adam(net.parameters(), lr=lr1)
    # set stopper
    stopper = earlyStopper(patience=3)
    # start traning
    iters = 0
    net.train()
    tbar = tqdm(range(num_epochs))
    elapsed, elapsed_val = 0, 0
    for epoch in tbar:
        tick = time.time()
        # multi data
        if type(dataloader) is tuple:
            loader = zip(*dataloader)
            loader_val = zip(*dataloader_val)
        else:
            loader = dataloader
            loader_val = dataloader_val
        # load data
        for data in loader:
            # convert tuple
            if type(dataloader) is tuple:
                # unzip data
                data = [torch.stack([d[0] for d in data]), # feat
                        torch.stack([d[1] for d in data]), # cost
                        torch.stack([d[2] for d in data]), # sol
                        torch.stack([d[3] for d in data])] # obj
            # cuda
            if next(net.parameters()).is_cuda:
                data = [d.cuda() for d in data]
            # forward pass
            loss = net(*data)
            # initialization
            if iters == 0:
                # init weights
                weights = torch.ones_like(loss)
                weights = torch.nn.Parameter(weights)
                T = weights.sum().detach() # sum of weights
                # set optimizer for weights
                optimizer2 = torch.optim.Adam([weights], lr=lr2)
                # set L(0)
                l0 = loss.detach()
            # compute the weighted loss
            weighted_loss = weights @ loss
            # clear gradients of network
            optimizer1.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            optimizer2.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()
            # log weights and loss
            weights_log.append(weights.detach().cpu().numpy().copy())
            loss_log.append(loss.detach().cpu().numpy().copy().sum())
            tbar.set_description("Loss: {:3.4f}".format(loss.sum().item()))
            # update model weights
            optimizer1.step()
            # update loss weights
            optimizer2.step()
            # renormalize weights
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer2 = torch.optim.Adam([weights], lr=lr2)
            # update iters
            iters += 1
        # time
        tock = time.time()
        elapsed += tock - tick
        # learning rate decay
        if lr_decay:
            if (epoch == int(num_epochs*0.6)) or (epoch == int(num_epochs*0.8)):
                for g in optimizer1.param_groups:
                    g['lr'] /= 10
        # early stop
        if earlystop:
            if epoch % val_step == 0:
                tick = time.time()
                loss = 0
                with torch.no_grad():
                    for data in loader_val:
                        # convert tuple
                        if type(dataloader_val) is tuple:
                            # unzip data
                            data = [torch.stack([d[0] for d in data]), # feat
                                    torch.stack([d[1] for d in data]), # cost
                                    torch.stack([d[2] for d in data]), # sol
                                    torch.stack([d[3] for d in data])] # obj
                        # cuda
                        if next(net.parameters()).is_cuda:
                            data = [d.cuda() for d in data]
                        # forward pass
                        loss += net(*data).sum()
                tock = time.time()
                elapsed_val += tock - tick
                if stopper.stop(loss):
                    break
    # get logs
    return np.stack(weights_log), loss_log, elapsed, elapsed_val
