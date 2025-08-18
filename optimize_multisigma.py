# optimize_multisigma.py

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from modules.BS_MixG_model import C_MixG_Torch_MultiSigma
from utils import FutureValue, loss_torch

class MultiSigmaModel(nn.Module):
    """一个封装了所有可训练参数的PyTorch模型"""
    def __init__(self, n, X, S_t):
        super(MultiSigmaModel, self).__init__()
        # 1. pi_raw: softmax前的pilogits，保证pi和为1且非负
        self.pi_raw = nn.Parameter(torch.ones(n + 1))
        
        # 2. mus: 初始化为log执行价格，更好的起点
        mu_init = np.log(np.append(X, S_t))
        if len(mu_init) > n + 1: mu_init = mu_init[:n+1]
        self.mus = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))

        # 3. sigmas_raw: log(sigmas)，保证sigma永远为正
        # 初始sigma设为0.1，log(0.1)约为-2.3
        self.sigmas_raw = nn.Parameter(torch.full((n + 1,), -2.3))

    @property
    def pis(self):
        return torch.softmax(self.pi_raw, dim=0)

    @property
    def sigmas(self):
        return torch.exp(self.sigmas_raw) # 保证sigma > 0

    def forward(self, X, r, tau):
        return C_MixG_Torch_MultiSigma(X, r, tau, self.sigmas, self.mus, self.pis)

    def get_no_arbitrage_terms(self, S_t, r, d, tau):
        # 计算约束的左右两边
        # Left term: E[S_T]
        left_term = torch.sum(self.pis * torch.exp(self.mus + self.sigmas**2 / 2))
        # Right term: F_t
        right_term = torch.tensor(FutureValue(S_t, r, d, tau))
        return left_term, right_term

def optimize_end_to_end(C_obs, n, X, S_t, r, tau, d, epochs=5000, lr=1e-3, constraint_weight=10.0):
    
    # 转换为Tensor
    X_torch = torch.tensor(X, dtype=torch.float32)
    C_obs_torch = torch.tensor(C_obs, dtype=torch.float32)
    r_torch = torch.tensor(r, dtype=torch.float32)
    tau_torch = torch.tensor(tau, dtype=torch.float32)
    d_torch = torch.tensor(d, dtype=torch.float32)

    # 初始化模型和优化器
    model = MultiSigmaModel(n, X, S_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("--- Starting End-to-End Optimization for Multi-Sigma Model ---")
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # 1. 计算价格拟合误差 (MSE Loss)
        C_pred = model(X_torch, r_torch, tau_torch)
        mse_loss = loss_torch(C_pred, C_obs_torch)
        
        # 2. 计算无套利约束惩罚
        left_term, right_term = model.get_no_arbitrage_terms(S_t, r, d, tau)
        constraint_loss = (left_term - right_term)**2
        
        # 3. 总损失 = 拟合误差 + 带权重的约束惩罚
        total_loss = mse_loss + constraint_weight * constraint_loss
        
        total_loss.backward()
        optimizer.step()

    print("Optimization finished.")
    print(f"Final MSE Loss: {mse_loss.item():.4f}")
    print(f"Final Constraint Difference: {(left_term - right_term).item():.4f}")

    # 返回优化后的参数 (转换为numpy数组)
    return {
        "pis": model.pis.detach().numpy(),
        "mus": model.mus.detach().numpy(),
        "sigmas": model.sigmas.detach().numpy()
    }