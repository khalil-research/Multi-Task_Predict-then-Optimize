#!/usr/bin/env python
# coding: utf-8
"""
Fully-connected Neural network
"""

import torch
from torch import nn

from pyepo.func import SPOPlus, perturbedFenchelYoung


class reg(nn.Module):
    """
    Linear layer model with softplus
    """
    def __init__(self, p, m):
        super(reg, self).__init__()
        self.linear = nn.Linear(p, m*(m-1)//2)
        self.softp = nn.Softplus(threshold=5)

    def forward(self, x):
        h = self.linear(x)
        out = self.softp(h)
        return out


class fcNet(nn.Module):
    """
    Full-connected prediction model
    """
    def __init__(self, p, m):
        super(fcNet, self).__init__()
        # layers
        self.fc1 = nn.Linear(p, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, m*(m-1)//2)
        self.softp = nn.Softplus(threshold=5)

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        out = self.softp(h)
        return out


class mtlSPO(nn.Module):
    """
    Multitask with SPO+ loss
    """
    def __init__(self, net, solvers, processes=1, mse=False):
        super(mtlSPO, self).__init__()
        # layers
        self.net = net
        # init SPO+ loss
        self.spop = nn.ModuleList([])
        for solver in solvers:
            self.spop.append(SPOPlus(solver, processes=processes))
        # mse flag
        self.mse = mse
        self.l2 = nn.MSELoss()

    def forward(self, x, c, w, z):
        cp = self.net(x)
        # compute loss
        loss = []
        for i, spop in enumerate(self.spop):
            # spo+
            loss.append(spop(cp, c, w[:,i], z[:,i]).mean())
        # mse
        if self.mse:
            loss.append(self.l2(c, cp))
        return torch.stack(loss)


class mtlPFYL(nn.Module):
    """
    Multitask with perturbed Fenchel-Young loss
    """
    def __init__(self, net, solvers, n_samples=1, epsilon=1.0, processes=1):
        super(mtlPFYL, self).__init__()
        # layers
        self.net = net
        # init SPO+ loss
        self.pfyl = nn.ModuleList([])
        for solver in solvers:
            self.pfyl.append(perturbedFenchelYoung(solver, n_samples=n_samples, epsilon=epsilon, processes=processes))

    def forward(self, x, c, w, z):
        cp = self.net(x)
        # compute loss
        loss = []
        for i, pfyl in enumerate(self.pfyl):
            # spo+
            loss.append(pfyl(cp, w[:,i]).mean())
        return torch.stack(loss)
