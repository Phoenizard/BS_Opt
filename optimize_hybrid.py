# optimize_hybrid.py

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from modules.BS_MixG_model import C_MixG_Torch_MultiSigma
from utils import FutureValue, loss_torch

class HybridSigmaModel(nn.Module):
    """'锚定-扰动'混合模型"""
    def __init__(self, n, X, S_t, sigma_best_anchor):
        super(HybridSigmaModel, self).__init__()
        self.pi_raw = nn.Parameter(torch.ones(n + 1))
        
        mu_init = np.log(np.append(X, S_t))
        if len(mu_init) > n + 1: mu_init = mu_init[:n+1]
        self.mus = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))

        # 存储锚点值
        self.sigma_best_anchor = torch.tensor(sigma_best_anchor, dtype=torch.float32)
        # 只学习微小的扰动项，初始化为0
        self.deltas = nn.Parameter(torch.zeros(n + 1))

    @property
    def pis(self):
        return torch.softmax(self.pi_raw, dim=0)

    @property
    def sigmas(self):
        # 核心创新点：最终的sigma是锚点和扰动项的结合
        return self.sigma_best_anchor * torch.exp(self.deltas)

    def forward(self, X, r, tau, sorted_indices):
        sorted_mus = self.mus[sorted_indices]
        sorted_sigmas = self.sigmas[sorted_indices]
        sorted_pis = self.pis[sorted_indices]
        return C_MixG_Torch_MultiSigma(X, r, tau, sorted_sigmas, sorted_mus, sorted_pis)

    def get_no_arbitrage_terms(self, S_t, r, d, tau):
        left_term = torch.sum(self.pis * torch.exp(self.mus + self.sigmas**2 / 2))
        right_term = torch.tensor(FutureValue(S_t, r, d, tau))
        return left_term, right_term

def optimize_hybrid_model(C_obs, n, X, S_t, r, tau, d, sigma_best_anchor, epochs=5000, lr=1e-3, constraint_weight=10.0, smoothness_weight=0.1):
    
    X_torch = torch.tensor(X, dtype=torch.float32)
    C_obs_torch = torch.tensor(C_obs, dtype=torch.float32)
    r_torch = torch.tensor(r, dtype=torch.float32)
    tau_torch = torch.tensor(tau, dtype=torch.float32)

    model = HybridSigmaModel(n, X, S_t, sigma_best_anchor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"--- Starting Hybrid Model Optimization (Anchor Sigma = {sigma_best_anchor:.4f}) ---")
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        sorted_indices = torch.argsort(model.mus)
        sorted_sigmas = model.sigmas[sorted_indices]
        
        C_pred = model(X_torch, r_torch, tau_torch, sorted_indices)
        mse_loss = loss_torch(C_pred, C_obs_torch)
        
        left_term, right_term = model.get_no_arbitrage_terms(S_t, r, d, tau)
        constraint_loss = (left_term - right_term)**2
        
        smoothness_loss = torch.sum((sorted_sigmas[1:] - sorted_sigmas[:-1])**2)
        
        # 扰动项正则化：鼓励扰动项delta保持较小的值
        delta_regularization = torch.sum(model.deltas**2)

        total_loss = mse_loss + constraint_weight * constraint_loss + smoothness_weight * smoothness_loss + 0.01 * delta_regularization
        
        total_loss.backward()
        optimizer.step()

    print("Optimization finished.")
    final_sorted_indices = torch.argsort(model.mus)
    return {
        "pis": model.pis[final_sorted_indices].detach().numpy(),
        "mus": model.mus[final_sorted_indices].detach().numpy(),
        "sigmas": model.sigmas[final_sorted_indices].detach().numpy()
    }