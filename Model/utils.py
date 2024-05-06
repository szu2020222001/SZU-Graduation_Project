#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 15:45
# @Author  : zhangyulin
# @File    : utils.py
# @Description :

import os
import h5py
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import os
import matplotlib.pyplot  as plt
from layers import GraphConvolution,Linear

def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

def normalize_A(A, symmetry=False,gaowei =False):
    A = F.relu(A)
    if symmetry:
        if gaowei:
            A = A + A.permute(0,2,1)
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            A = A + torch.transpose(A,0,1)
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


