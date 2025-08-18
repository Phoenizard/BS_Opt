import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd.scipy.stats import norm
import torch
from torch.distributions.normal import Normal

def F(X):
    return 1 - norm.cdf(X)

def F_autograd(X):
    return 1 - norm.cdf(X)

def C_MixG(X, r, tau, sigma, mu, pi) -> np.ndarray:
    """
    r: Risk-free rate
    d: Dividend yield
    tau: Time to expiration (in years)
    sigma: Volatility
    mu: Mean of the lognormal distribution
    X: Strike prices (can be a vector)
    Calculate the theoretical value of call options in G Mixtures model
    """

    # Ensure X is a numpy array for vectorized operations
    X = np.array(X)
    
    # Calculate d1 and d2 as vectors
    d1 = np.exp(-r * tau + mu + 0.5 * sigma ** 2)[:, np.newaxis] * F((np.log(X) - (mu[:, np.newaxis] + sigma ** 2)) / sigma)
    d2 = np.exp(-r * tau) * X * F((np.log(X) - mu[:, np.newaxis]) / sigma)
    
    # Sum over the mixture components
    C = np.sum((d1 - d2) * pi[:, np.newaxis], axis=0)
    
    return C

def F_torch(X):
    normal = Normal(0, 1)
    return 1 - normal.cdf(X)

def C_MixG_Torch(X, r, tau, sigma, mu, pi):
    """
    r: 风险无息利率
    tau: 到期时间（年）
    sigma: 波动率
    mu: 对数正态分布的均值
    X: 行权价格（可以是向量）
    pi: 混合模型的权重
    计算G混合模型中看涨期权的理论价值
    """
    if not isinstance(mu, torch.Tensor):
        print(type(mu))
        raise ValueError("mu should be a torch.Tensor")
    # 计算d1和d2为向量
    d1 = (torch.exp(-r * tau + mu + 0.5 * sigma ** 2)[:, None] * 
          F_torch((torch.log(X) - (mu[:, None] + sigma ** 2)) / sigma))
    d2 = (torch.exp(-r * tau) * X * 
          F_torch((torch.log(X) - mu[:, None]) / sigma))
    
    # 对混合组件求和
    C = torch.sum((d1 - d2) * pi[:, None], dim=0)
    return C


# 使用 autograd.numpy 改写的 C_MixG 函数
def C_MixG_autograd(X, r, tau, sigma, mu, pi) -> anp.ndarray:
    """
    r: Risk-free rate
    d: Dividend yield
    tau: Time to expiration (in years)
    sigma: Volatility
    mu: Mean of the lognormal distribution
    X: Strike prices (can be a vector)
    Calculate the theoretical value of call options in G Mixtures model
    """
    
    # Ensure X is an autograd.numpy array for vectorized operations
    X = anp.array(X)
    
    # Calculate d1 and d2 as vectors, using autograd.numpy for operations
    d1 = anp.exp(-r * tau + mu + 0.5 * sigma ** 2)[:, anp.newaxis] * F((anp.log(X) - (mu[:, anp.newaxis] + sigma ** 2)) / sigma)
    d2 = anp.exp(-r * tau) * X * F_autograd((anp.log(X) - mu[:, anp.newaxis]) / sigma)
    
    # Sum over the mixture components
    C = anp.sum((d1 - d2) * pi[:, anp.newaxis], axis=0)
    
    return C


if __name__ == '__main__':
    # Test the function
    r = 0.045
    d = 0.025
    tau = 30 / 365
    sigma = 0.2
    mu_init = np.random.uniform(low=7.107, high=7.265, size=26)
    # Convert to numpy array
    pi_init = np.ones(26) / 26
    X = np.linspace(1000, 1700, 25)
    C_pred = (C_MixG(X, r, tau, sigma, mu_init, pi_init))
    mu = torch.tensor(mu_init, dtype=torch.float32, requires_grad=True)
    C_pred_torch = C_MixG_Torch(X, r, tau, sigma, mu, pi_init)
    print(C_pred_torch)



# --- 请将以下函数添加到 modules/BS_MixG_model.py 文件末尾 ---

def C_MixG_MultiSigma(X, r, tau, sigmas, mus, pis):
    """
    Numpy-based MixG model that accepts a vector of sigmas.
    """
    X = np.asarray(X)
    sigmas = np.asarray(sigmas)
    mus = np.asarray(mus)
    pis = np.asarray(pis)

    # Broadcasting shapes: X(n,), mus(k,), sigmas(k,), pis(k,) -> C_pred(k, n)
    d1 = (-np.log(X) + mus[:, np.newaxis] + sigmas[:, np.newaxis]**2) / sigmas[:, np.newaxis]
    d2 = d1 - sigmas[:, np.newaxis]

    # Standard normal CDF
    C_pred_components = np.exp(-r * tau) * \
                        (np.exp(mus[:, np.newaxis] + sigmas[:, np.newaxis]**2 / 2) * norm.cdf(d1) - \
                         X[np.newaxis, :] * norm.cdf(d2))
    
    # Final price is the weighted sum
    C_final = np.sum(pis[:, np.newaxis] * C_pred_components, axis=0)
    return C_final

def C_MixG_Torch_MultiSigma(X, r, tau, sigmas, mus, pis):
    """
    PyTorch-based MixG model that accepts a vector of sigmas.
    """
    normal = torch.distributions.Normal(0, 1)

    # Broadcasting shapes: X(n,), mus(k,), sigmas(k,), pis(k,) -> C_pred(k, n)
    d1 = (-torch.log(X) + mus.unsqueeze(1) + sigmas.unsqueeze(1)**2) / sigmas.unsqueeze(1)
    d2 = d1 - sigmas.unsqueeze(1)
    
    C_pred_components = torch.exp(-r * tau) * \
                        (torch.exp(mus.unsqueeze(1) + sigmas.unsqueeze(1)**2 / 2) * normal.cdf(d1) - \
                         X.unsqueeze(0) * normal.cdf(d2))

    C_final = torch.sum(pis.unsqueeze(1) * C_pred_components, dim=0)
    return C_final