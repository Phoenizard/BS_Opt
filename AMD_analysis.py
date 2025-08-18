# Amd_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# --- 初始设置 ---
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from optimize import simulation_LOOCV
from modules.BS_MixG_model import C_MixG
from utils import loss 

def main():
    # --- 1. 加载和预处理数据 ---
    print("--- 1. Loading and processing AMD option data... ---")
    try:
        path = 'data/real_2018_6_7_AMD_data.csv'
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{path}'. Please ensure the path is correct.")
        return

    tau_values = np.sort(np.unique(data['tau(T/365)']))
    train_data = data[data['tau(T/365)'] == tau_values[0]].copy()
    test_data = data[data['tau(T/365)'] == tau_values[1]].copy()
    train_data['r'] = train_data['r'] / 100
    test_data['r'] = test_data['r'] / 100
    print("Data loaded and split into training and testing sets successfully.\n")

    # --- 2. 准备训练所需的核心变量 ---
    print("--- 2. Preparing variables for model training... ---")
    X_train = train_data['strick_price(X)'].values
    C_obs_train = train_data['option_price(C_bid)'].values
    n_train = len(C_obs_train)
    r_train = train_data['r'].iloc[0]
    tau_train = train_data['tau(T/365)'].iloc[0]
    S_t_train = train_data['close'].iloc[0]
    print(f"Training data size (n): {n_train}")
    print(f"Spot Price (S_t): {S_t_train}")
    print(f"Interest Rate (r): {r_train:.4f}")
    print(f"Time to Maturity (tau): {tau_train:.4f}\n")

    # --- 3. 执行LOOCV以寻找最优sigma并训练最终模型 ---
    sigma_grid = np.linspace(0.06, 0.07, 20) 

    # 【已修正】接收5个返回值，修复ValueError
    C_fitted, mu_final, pi_final, best_sigma, loocv_errors = simulation_LOOCV(
        C_obs_full=C_obs_train,
        n_full=n_train,
        X_full=X_train,
        S_t=S_t_train,
        r=r_train,
        tau=tau_train,
        d=0,
        sigma_grid=sigma_grid
    )
    
    # --- 4. 可视化LOOCV误差 ---
    print("\n--- 4. Visualizing LOOCV Errors vs. Sigma... ---")
    plt.figure(figsize=(10, 6))
    valid_results = [(s, e) for s, e in zip(sigma_grid, loocv_errors) if np.isfinite(e)]
    if valid_results:
        valid_sigmas, valid_errors_plot = zip(*valid_results)
        plt.plot(valid_sigmas, valid_errors_plot, 'o-', label='LOOCV Error')
    plt.axvline(x=best_sigma, color='r', linestyle='--', label=f'Optimal Sigma = {best_sigma:.4f}')
    plt.title('LOOCV Error vs. Sigma')
    plt.xlabel('Sigma Value')
    plt.ylabel('Total LOOCV Squared Error (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 5. 可视化训练集的拟合结果 ---
    print("\n--- 5. Visualizing training set results... ---")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, C_obs_train, label='Observed C (Train)', c='blue', alpha=0.7)
    plt.plot(X_train, C_fitted, 'r-o', markersize=4, label=f'Fitted C (Train) with best sigma={best_sigma:.4f}')
    plt.xlabel('Strike Price (X)')
    plt.ylabel('Option Price (C)')
    plt.title('Final Fit on AMD Training Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 6. 在测试集上进行预测与评估 ---
    print("\n--- 6. Evaluating model on the test set... ---")
    X_test = test_data['strick_price(X)'].values
    C_obs_test = test_data['option_price(C_bid)'].values
    tau_test = test_data['tau(T/365)'].iloc[0]
    r_test = test_data['r'].iloc[0]
    C_pred_test = C_MixG(X_test, r_test, tau_test, best_sigma, mu_final, pi_final)
    test_mse = loss(C_pred_test, C_obs_test)
    print(f"Mean Squared Error (MSE) on Test Set with optimal sigma: {test_mse:.4f}")

    # --- 7. 可视化测试集的预测结果 ---
    print("\n--- 7. Visualizing test set results... ---")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, C_obs_test, label='Observed C (Test)', c='blue', alpha=0.7)
    plt.plot(X_test, C_pred_test, 'g-o', markersize=4, label='Predicted C (Test)')
    plt.xlabel('Strike Price (X)')
    plt.ylabel('Option Price (C)')
    plt.title('Prediction on AMD Test Data with Optimal Sigma')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- 8. 计算并绘制状态价格密度 (SPD) ---
    print("\n--- 8. Calculating and plotting State-Price Density (SPD)... ---")
    try:
        import torch
        from modules.BS_MixG_model import C_MixG_Torch
        mu_opt_torch = torch.tensor(mu_final, dtype=torch.float32)
        pi_opt_torch = torch.tensor(pi_final, dtype=torch.float32)
        sigma_torch = torch.tensor(best_sigma, dtype=torch.float32)
        r_torch = torch.tensor(r_train, dtype=torch.float32)
        tau_torch = torch.tensor(tau_train, dtype=torch.float32)
        X_dense = torch.linspace(X_train.min() * 0.8, X_train.max() * 1.2, 200, requires_grad=True)
        C_dense = C_MixG_Torch(X_dense, r_torch, tau_torch, sigma_torch, mu_opt_torch, pi_opt_torch)
        first_derivative = torch.autograd.grad(C_dense.sum(), X_dense, create_graph=True)[0]
        second_derivative = torch.autograd.grad(first_derivative.sum(), X_dense)[0]
        spd = torch.exp(r_torch * tau_torch) * second_derivative
        plt.figure(figsize=(10, 6))
        plt.plot(X_dense.detach().numpy(), spd.detach().numpy(), label='Estimated State-Price Density (SPD)')
        plt.title('State-Price Density Estimated from AMD Option Prices')
        plt.xlabel('Future Asset Price (S_T)')
        plt.ylabel('Density')
        plt.ylim(bottom=0, top=np.percentile(spd.detach().numpy(), 99.5))
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("PyTorch is not installed. Skipping SPD plot generation.")
    except Exception as e:
        print(f"An error occurred during SPD plot generation: {e}")

if __name__ == "__main__":
    main()