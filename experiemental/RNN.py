# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:05:53 2020

@author: Edwin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3,3)
inputs = [torch.randn(1,3) for _ in range(5)]

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

for i in inputs:
    # Step through the sequence on element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    