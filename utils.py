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

# Function to simulate ST using the Black-Scholes model and estimate the standard deviation of ln(ST)
def simulate_and_estimate_std(S, K, T, r, d, sigma, num_simulations=10):
    std_estimates = []

    for _ in range(num_simulations):
        Z = np.random.normal(size=len(K))
        ST = S * np.exp((r - d - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        ln_ST = np.log(ST)
        _, std_estimate = norm.fit(ln_ST)
        std_estimates.append(std_estimate)

    return std_estimates

def constraint_loss(left_term, right_term, weight=1.0):
    return weight * torch.mean((left_term - right_term) ** 2)
