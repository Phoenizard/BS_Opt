# Amd_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime

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
    fig_loocv, ax_loocv = plt.subplots(figsize=(10, 6))
    valid_results = [(s, e) for s, e in zip(sigma_grid, loocv_errors) if np.isfinite(e)]
    if valid_results:
        valid_sigmas, valid_errors_plot = zip(*valid_results)
        ax_loocv.plot(valid_sigmas, valid_errors_plot, 'o-', label='LOOCV Error')
    ax_loocv.axvline(x=best_sigma, color='r', linestyle='--', label=f'Optimal Sigma = {best_sigma:.4f}')
    ax_loocv.set_title('LOOCV Error vs. Sigma')
    ax_loocv.set_xlabel('Sigma Value')
    ax_loocv.set_ylabel('Total LOOCV Squared Error (log scale)')
    ax_loocv.set_yscale('log')
    ax_loocv.legend()
    ax_loocv.grid(True)
    plt.show()

    # --- 5. 可视化训练集的拟合结果 ---
    print("\n--- 5. Visualizing training set results... ---")
    fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
    ax_fit.scatter(X_train, C_obs_train, label='Observed C (Train)', c='blue', alpha=0.7)
    ax_fit.plot(X_train, C_fitted, 'r-o', markersize=4, label=f'Fitted C (Train) with best sigma={best_sigma:.4f}')
    ax_fit.set_xlabel('Strike Price (X)')
    ax_fit.set_ylabel('Option Price (C)')
    ax_fit.set_title('Final Fit on AMD Training Data')
    ax_fit.legend()
    ax_fit.grid(True)
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
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    ax_pred.scatter(X_test, C_obs_test, label='Observed C (Test)', c='blue', alpha=0.7)
    ax_pred.plot(X_test, C_pred_test, 'g-o', markersize=4, label='Predicted C (Test)')
    ax_pred.set_xlabel('Strike Price (X)')
    ax_pred.set_ylabel('Option Price (C)')
    ax_pred.set_title('Prediction on AMD Test Data with Optimal Sigma')
    ax_pred.legend()
    ax_pred.grid(True)
    plt.show()
    
    # --- 8. 计算并绘制状态价格密度 (SPD) ---
    print("\n--- 8. Calculating and plotting State-Price Density (SPD)... ---")
    fig_spd = None
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
        
        fig_spd, ax_spd = plt.subplots(figsize=(10, 6))
        ax_spd.plot(X_dense.detach().numpy(), spd.detach().numpy(), label='Estimated State-Price Density (SPD)')
        ax_spd.set_title('State-Price Density Estimated from AMD Option Prices')
        ax_spd.set_xlabel('Future Asset Price (S_T)')
        ax_spd.set_ylabel('Density')
        ax_spd.set_ylim(bottom=0, top=np.percentile(spd.detach().numpy(), 99.5))
        ax_spd.legend()
        ax_spd.grid(True)
        plt.show()
    except ImportError:
        print("PyTorch is not installed. Skipping SPD plot generation.")
    except Exception as e:
        print(f"An error occurred during SPD plot generation: {e}")

    # --- 9. 【新增】保存结果 ---
    print("\n--- 9. Saving results to file... ---")
    # 创建一个带时间戳的文件夹来存放本次运行的结果
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 保存参数
    np.savez(os.path.join(output_dir, "optimized_parameters.npz"), 
             mu=mu_final, 
             pi=pi_final, 
             best_sigma=best_sigma)

    # 保存结果摘要
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"--- Summary of Run on {timestamp} ---\n")
        f.write(f"Optimal Sigma: {best_sigma:.6f}\n")
        f.write(f"Test Set MSE: {test_mse:.6f}\n")
        f.write("\n--- Optimized Parameters ---\n")
        f.write("Mus:\n")
        f.write(np.array2string(mu_final, precision=4))
        f.write("\n\nPis:\n")
        f.write(np.array2string(pi_final, precision=4))

    # 保存图像
    fig_loocv.savefig(os.path.join(output_dir, "loocv_error_plot.png"))
    fig_fit.savefig(os.path.join(output_dir, "training_fit_plot.png"))
    fig_pred.savefig(os.path.join(output_dir, "test_prediction_plot.png"))
    if fig_spd:
        fig_spd.savefig(os.path.join(output_dir, "spd_plot.png"))
    
    print("All results have been saved successfully.")


if __name__ == "__main__":
    main()