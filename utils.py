# utils.py

import numpy as np
import torch
from scipy.stats import norm

def loss_torch(C_pred, C_true):
    if C_pred.shape != C_true.shape:
        raise ValueError("C_pred and C_true should have the same shape")
    return torch.mean((C_pred - C_true) ** 2)

def loss(C_pred, C_true):
    if C_pred.shape != C_true.shape:
        raise ValueError("C_pred and C_true should have the same shape")
    return np.mean((C_pred - C_true) ** 2)

def Expection(mu, pi, sigma_0):
    """计算期望值 E[S_T]"""
    return np.dot(np.exp(mu), pi) * np.exp(sigma_0 ** 2 / 2)

def FutureValue(S_t, r, d, tau):
    """计算远期价格 F_t"""
    return S_t * np.exp((r - d) * tau)