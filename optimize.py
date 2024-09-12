import numpy as np
from modules.BS_MixG_model import C_MixG
from modules.BS_Theoretical_Model import BS_Theoretical_Value, add_noise_to_option_values
from matplotlib import pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from modules.BS_MixG_model import C_MixG_Torch
import torch
from utils import loss_torch, Expection, FutureValue, loss
import warnings
warnings.filterwarnings('ignore')

# 目标函数
def objective(pi, X, r, tau, sigma_0, mu, C_obs):
    # C_pred 通过 C_MixG 计算
    C_pred = C_MixG(X, r, tau, sigma_0, mu, pi)
    return loss(C_pred, C_obs)

def constraint_expection(pi, mu, sigma_0, FV):
    # 计算期望值和未来价值
    E_value = Expection(mu, pi, sigma_0)
    return E_value - FV

def Quad_Optimize(X, r, tau, sigma_0, mu, pi_init, C_obs, FV, n):
    epsilon = 1e-6
    bounds = [(epsilon, 1 - epsilon)] * (n + 1)
    expection_constraint = {'type': 'eq', 'fun': constraint_expection, 'args': (mu, sigma_0, FV)}
    linear_constraint = LinearConstraint(np.ones(n + 1), 1, 1)
    res = minimize(objective, x0=pi_init, args=(X, r, tau, sigma_0, mu, C_obs), method='SLSQP', 
                constraints=[expection_constraint, linear_constraint], bounds=bounds,
                options={'disp': False, 'ftol': 1e-9})
    return res

def gradient_hessian(X, r, tau, sigma_0, mu_torch, pi, C_obs, n):
    # Calculate the value of call options in G Mixtures model
    if not isinstance(mu_torch, torch.Tensor) or not isinstance(C_obs, torch.Tensor) or not isinstance(X, torch.Tensor):
        raise ValueError("mu_torch, C_obs, and X should be torch.Tensor")
    if mu_torch.requires_grad is False:
        raise ValueError("mu_torch should require grad")
    C_pred = C_MixG_Torch(X, r, tau, sigma_0, mu_torch, pi)
    loss = loss_torch(C_pred, C_obs)

    # 计算 loss 相对于 mu_torch 的梯度
    grad = torch.autograd.grad(loss, mu_torch, create_graph=True)[0]
    hessian_matrix = torch.zeros(n + 1, n + 1)
    # 计算 Hessian，每个元素是梯度对 mu_torch 的二次导数
    for i in range(n + 1):
        grad_i = grad[i]
        hessian_row = torch.autograd.grad(grad_i, mu_torch, retain_graph=True)[0]
        hessian_matrix[i] = hessian_row
    # print("Hessian Matrix: ", hessian_matrix)
    # IF The diagonal element is zero, add a small value to the diagonal element
    for i in range(n + 1):
        if hessian_matrix[i, i] == 0:
            hessian_matrix[i, i] = 1e-9
    return grad, hessian_matrix

def simulation_LOOCV(sigma_0, C_obs, n, X, r, tau,d, mu_guess=None):
    # 优化参数
    mu = np.random.uniform(low=7.107, high=7.265, size=(n + 1))
    if mu_guess is not None:
        # n+1的向量
        mu = np.random.uniform(low=mu_guess, high=mu_guess + 0.1, size=(n + 1))
    pi_init = np.ones(n + 1) / (n + 1)  # 初始猜测的 pi
    C_pred = C_MixG(X, r, tau, sigma_0, mu, pi_init)
    #=======================二次优化求解器=======================
    FV = FutureValue(r, d, tau, X)
    res = Quad_Optimize(X, r, tau, sigma_0, mu, pi_init, C_obs, FV, n)
    pi_opt = res.x
    loss_opt_1 = res.fun
    C_pred_opt_1 = C_MixG(X, r, tau, sigma_0, mu, pi_opt)
    # print("Optimized loss after Qrad_Opt: ", loss_opt_1)
    # print("Optimized Expection: ", Expection(mu, pi_opt), "Future Value: ", FV)
    #=======================Netwon-Raphson求解器=================
    X_torch = torch.tensor(X, dtype=torch.float32)
    C_obs_torch = torch.tensor(C_obs, dtype=torch.float32)
    mu_opt = torch.tensor(mu, dtype=torch.float32, requires_grad=True)
    pi_opt = torch.tensor(pi_opt, dtype=torch.float32)
    r_torch = torch.tensor(r, dtype=torch.float32)
    epoch = 5
    for i in range(epoch):
        grad, hessian = gradient_hessian(X_torch, r_torch, tau, sigma_0, mu_opt, pi_opt, C_obs_torch, n)
        mu_opt = mu_opt - torch.inverse(hessian) @ grad
    mu_opt = mu_opt.detach().numpy()
    pi_opt = pi_opt.detach().numpy()
    # Calculate the Expection and Future Value
    C_pred_opt_2 = C_MixG(X, r, tau, sigma_0, mu_opt, pi_opt)
    loss_opt_2 = loss(C_pred_opt_2, C_obs)
    print("Loss after Opt : ", loss_opt_2)
    # 绘制图像BS和混合模型的对比
    return C_pred, C_pred_opt_1, C_pred_opt_2, loss_opt_2, mu_opt, pi_opt
    