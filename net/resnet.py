#!/usr/bin/env python
# coding: utf-8
"""
Residual convolutional neural network
"""

import torch
from torch import nn
from torchvision.models import resnet18

from pyepo.func import SPOPlus, perturbedFenchelYoung


class partialResNetSeperated(nn.Module):
    """
    Mutiple seperated truncated ResNet18
    """
    def __init__(self, k, n_tasks):
        super(partialResNetSeperated, self).__init__()
        self.n_tasks = n_tasks
        # towers layer
        self.towers = nn.ModuleList([])
        for _ in range(n_tasks):
            # init resnet 18
            resnet = resnet18(pretrained=False)
            # first five layers of ResNet18 as shared layer
            conv1 = resnet.conv1
            bn = resnet.bn1
            relu = resnet.relu
            maxpool1 = resnet.maxpool
            blocks = resnet.layer1
            # conv to 1 channel
            conv2  = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # max pooling
            maxpool2 = nn.AdaptiveMaxPool2d((k,k))
            # tower
            tower = nn.Sequential(conv1, bn, relu, maxpool1, blocks, conv2, maxpool2)
            self.towers.append(tower)

    def forward(self, x):
        outs = []
        for tower in self.towers:
            out = tower(x)
            # reshape for optmodel
            out = torch.squeeze(out, 1)
            out = out.reshape(out.shape[0], -1)
            outs.append(out)
        return torch.stack(outs)


class partialResNetMTL(nn.Module):
    """
    Truncated ResNet18 with multiple towers
    """
    def __init__(self, k, n_tasks):
        super(partialResNetMTL, self).__init__()
        self.n_tasks = n_tasks
        # init resnet 18
        resnet = resnet18(pretrained=False)
        # first five layers of ResNet18 as shared layer
        self.conv = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.block = resnet.layer1[0]
        # towers layer
        self.towers = nn.ModuleList([])
        for _ in range(n_tasks):
            # basic block
            block = resnet18(pretrained=False).layer1[1]
            # conv to 1 channel
            conv  = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # max pooling
            maxpool = nn.AdaptiveMaxPool2d((k,k))
            # tower
            tower = nn.Sequential(block, conv, maxpool)
            self.towers.append(tower)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.block(h)
        outs = []
        for tower in self.towers:
            out = tower(h)
            # reshape for optmodel
            out = torch.squeeze(out, 1)
            out = out.reshape(out.shape[0], -1)
            outs.append(out)
        return torch.stack(outs)



class partialResNetMSE(nn.Module):
    """
    Truncated ResNet18 with MSE for 2-stage
    """
    def __init__(self, nnet):
        super(partialResNetMSE, self).__init__()
        self.nnet = nnet
        self.n_tasks = nnet.n_tasks
        self.l2 = nn.MSELoss()

    def forward(self, xs, cs, ws, zs):
        loss = []
        for i in range(self.n_tasks):
            # predict
            cp = self.nnet(xs[i])
            # mse
            loss.append(self.l2(cp[i], cs[i]))
        return torch.stack(loss)


class partialResNetSPO(nn.Module):
    """
    Truncated ResNet18 with multiple SPO+
    """
    def __init__(self, nnet, solver, processes=1, mse=False):
        super(partialResNetSPO, self).__init__()
        self.nnet = nnet
        self.n_tasks = nnet.n_tasks
        self.spop = SPOPlus(solver, processes=processes)
        # mse flag
        self.mse = mse
        self.l2 = nn.MSELoss()

    def forward(self, xs, cs, ws, zs):
        loss = []
        for i in range(self.n_tasks):
            # predict
            cp = self.nnet(xs[i])
            # spo+
            loss.append(self.spop(cp[i], cs[i], ws[i], zs[i]).mean())
            # mse
            if self.mse:
                loss.append(self.l2(cs[i], cp[i]))
        return torch.stack(loss)


class partialResNetPFYL(nn.Module):
    """
    Truncated ResNet18 with multiple PFYL
    """
    def __init__(self, nnet, solver, n_samples=1, epsilon=1.0, processes=1):
        super(partialResNetPFYL, self).__init__()
        self.nnet = nnet
        self.n_tasks = nnet.n_tasks
        self.pfyl = perturbedFenchelYoung(solver, n_samples=n_samples, epsilon=epsilon, processes=processes)

    def forward(self, xs, cs, ws, zs):
        loss = []
        for i in range(self.n_tasks):
            # predict
            cp = self.nnet(xs[i])
            # spo+
            loss.append(self.pfyl(cp[i], ws[i]).mean())
        return torch.stack(loss)
