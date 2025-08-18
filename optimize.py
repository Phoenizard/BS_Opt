# optimize.py

import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint
from modules.BS_MixG_model import C_MixG, C_MixG_Torch
from utils import loss_torch, loss, Expection, FutureValue
import warnings
warnings.filterwarnings('ignore')

# --- 内部使用的辅助函数 ---

C_obs_global = None # 临时全局变量

def _objective(pi, X, r, tau, sigma, mu):
    C_pred = C_MixG(X, r, tau, sigma, mu, pi)
    return loss(C_pred, C_obs_global)

def _constraint_expection(pi, mu, sigma, FV):
    E_value = Expection(mu, pi, sigma)
    return E_value - FV

def _quad_optimize(X, r, tau, sigma, mu, pi_init, C_obs, FV, n):
    global C_obs_global
    C_obs_global = C_obs
    epsilon = 1e-6
    bounds = [(epsilon, 1 - epsilon)] * (n + 1)
    constraints = [
        {'type': 'eq', 'fun': _constraint_expection, 'args': (mu, sigma, FV)},
        LinearConstraint(np.ones(n + 1), 1, 1)
    ]
    res = minimize(_objective, x0=pi_init, args=(X, r, tau, sigma, mu), method='SLSQP',
                   constraints=constraints, bounds=bounds, options={'disp': False, 'ftol': 1e-9})
    return res.x

def _gradient_hessian(X, r, tau, sigma, mu_torch, pi, C_obs, n):
    C_pred = C_MixG_Torch(X, r, tau, sigma, mu_torch, pi)
    loss_val = loss_torch(C_pred, C_obs)
    grad = torch.autograd.grad(loss_val, mu_torch, create_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(mu_torch), torch.eye(n + 1)
    hessian_matrix = torch.zeros(n + 1, n + 1)
    for i in range(n + 1):
        grad_i = grad[i]
        hessian_row = torch.autograd.grad(grad_i, mu_torch, retain_graph=True, allow_unused=True)[0]
        if hessian_row is not None:
            hessian_matrix[i] = hessian_row
    return grad, hessian_matrix

def _newton_optimize(X, C_obs, r, tau, sigma, mu_init, pi_opt, n):
    X_torch = torch.tensor(X, dtype=torch.float32)
    C_obs_torch = torch.tensor(C_obs, dtype=torch.float32)
    mu_opt_torch = torch.tensor(mu_init, dtype=torch.float32, requires_grad=True)
    pi_opt_torch = torch.tensor(pi_opt, dtype=torch.float32)
    r_torch = torch.tensor(r, dtype=torch.float32)
    regularization_strength = 1e-6
    for _ in range(5):
        grad, hessian = _gradient_hessian(X_torch, r_torch, tau, sigma, mu_opt_torch, pi_opt_torch, C_obs_torch, n)
        try:
            regularized_hessian = hessian + torch.eye(n + 1) * regularization_strength
            delta = torch.linalg.solve(regularized_hessian, grad)
            mu_opt_torch.data -= delta
        except torch.linalg.LinAlgError:
            mu_opt_torch.data -= regularization_strength * grad.data
    return mu_opt_torch.detach().numpy()

# --- 主功能函数 ---

def simulation_LOOCV(C_obs_full, n_full, X_full, S_t, r, tau, d, sigma_grid):
    loocv_errors = []

    def _train_single_model(sigma, C_obs, n, X, S_t, r, tau, d):
        mu_init = np.log(np.append(X, S_t))
        if len(mu_init) > n + 1: mu_init = mu_init[:n+1]
        pi_init = np.ones(n + 1) / (n + 1)
        FV = FutureValue(S_t, r, d, tau)
        pi_opt = _quad_optimize(X, r, tau, sigma, mu_init, pi_init, C_obs, FV, n)
        mu_opt = _newton_optimize(X, C_obs, r, tau, sigma, mu_init, pi_opt, n)
        return mu_opt, pi_opt

    print("Starting LOOCV to find the best sigma...")
    for sigma_val in sigma_grid:
        total_squared_error = 0
        print(f"--- Testing sigma = {sigma_val:.4f} ---")
        for i in range(n_full):
            try:
                X_val, C_val = X_full[i], C_obs_full[i]
                X_train = np.delete(X_full, i)
                C_train = np.delete(C_obs_full, i)
                n_train = n_full - 1
                mu_opt_loo, pi_opt_loo = _train_single_model(sigma_val, C_train, n_train, X_train, S_t, r, tau, d)
                if np.isnan(mu_opt_loo).any() or np.isnan(pi_opt_loo).any():
                    raise ValueError("Optimization resulted in NaN values.")
                predicted_C = C_MixG(np.array([X_val]), r, tau, sigma_val, mu_opt_loo, pi_opt_loo)[0]
                squared_error = (predicted_C - C_val) ** 2
                total_squared_error += squared_error
            except Exception as e:
                print(f"  Warning: LOOCV failed at iteration {i+1} for sigma={sigma_val}. Error: {e}")
                total_squared_error = np.inf
                break
        loocv_errors.append(total_squared_error)
        print(f"Total LOOCV error for sigma = {sigma_val:.4f} is {total_squared_error:.4f}\n")

    valid_errors = np.array([e for e in loocv_errors if np.isfinite(e)])
    if len(valid_errors) == 0:
        raise ValueError("LOOCV failed for all sigma values.")
    best_sigma_index = np.argmin(loocv_errors)
    best_sigma = sigma_grid[best_sigma_index]
    print(f"LOOCV finished. Optimal sigma found: {best_sigma}\n")

    print("--- Training final model with optimal sigma on the full dataset... ---")
    mu_opt_final, pi_opt_final = _train_single_model(best_sigma, C_obs_full, n_full, X_full, S_t, r, tau, d)
    C_pred_final = C_MixG(X_full, r, tau, best_sigma, mu_opt_final, pi_opt_final)
    final_loss = loss(C_pred_final, C_obs_full)
    print(f"Final model training complete. MSE on full dataset: {final_loss:.4f}")
    
    # 【已修正】返回5个值，修复ValueError
    return C_pred_final, mu_opt_final, pi_opt_final, best_sigma, loocv_errors