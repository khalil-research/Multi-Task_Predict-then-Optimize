#!/usr/bin/env python
# coding: utf-8
"""
Early stopping
"""

import numpy as np

class earlyStopper:
    """
    Early stopping for training
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.min_loss = np.inf

    def stop(self, loss):
        if loss + 1e-5 < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
