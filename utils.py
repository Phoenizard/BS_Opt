import numpy as np
import torch
from scipy.stats import norm

def phi(x):
    return norm.pdf(x)

def F(X):
    return 1 - norm.cdf(X)

def loss_torch(C_pred, C_true):
    if C_pred.shape != C_true.shape:
        raise ValueError("C_pred and C_true should have the same shape")
    return torch.mean((C_pred - C_true) ** 2)

def loss(C_pred, C_true):
    if C_pred.shape != C_true.shape:
        raise ValueError("C_pred and C_true should have the same shape")
    return np.mean((C_pred - C_true) ** 2)

# 计算Expection价值
def Expection(mu, pi, sigma_0=0.2):
    return np.dot(np.exp(mu), pi) * np.exp(sigma_0 ** 2 / 2)

# 计算看涨期权下未来价值
def FutureValue(r, d, tau, X):
    return (np.exp((r - d) * tau) * X).mean()