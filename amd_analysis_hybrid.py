# amd_analysis_bayesopt.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- 初始设置 ---
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入您项目中的函数
from optimize_hybrid import optimize_hybrid_model
from modules.BS_MixG_model import C_MixG_MultiSigma
from utils import loss

def main():
    # --- 1. 数据加载与处理 ---
    print("--- 1. Loading and processing data... ---")
    try:
        path = 'data/real_2018_6_7_AMD_data.csv'
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{path}'. Please ensure the path is correct.")
        return

    tau_values = np.sort(np.unique(data['tau(T/365)']))
    train_data_full = data[data['tau(T/365)'] == tau_values[0]].copy()
    test_data = data[data['tau(T/365)'] == tau_values[1]].copy()
    train_data_full['r'] = train_data_full['r'] / 100
    test_data['r'] = test_data['r'] / 100

    X_full_train = train_data_full['strick_price(X)'].values
    C_obs_full_train = train_data_full['option_price(C_bid)'].values
    S_t_full_train = train_data_full['close'].iloc[0]
    r_full_train = train_data_full['r'].iloc[0]
    tau_full_train = train_data_full['tau(T/365)'].iloc[0]

    # 从训练集中分割出验证集
    X_train_sub, X_val, C_train_sub, C_val = train_test_split(
        X_full_train, C_obs_full_train, test_size=0.2, random_state=42
    )
    n_train_sub = len(X_train_sub)
    print("Data loaded and split successfully.\n")

    # --- 2. 定义贝叶斯优化的搜索空间和目标函数 ---
    space  = [Real(1e-5, 1.0, "log-uniform", name='smoothness_weight')]

    @use_named_args(space)
    def objective(**params):
        weight = params['smoothness_weight']
        print(f"\n---> Testing smoothness_weight = {weight:.6f}")
        
        trained_params = optimize_hybrid_model(
            C_obs=C_train_sub, n=n_train_sub, X=X_train_sub, S_t=S_t_full_train,
            r=r_full_train, tau=tau_full_train, d=0, 
            sigma_best_anchor=0.0647,
            smoothness_weight=weight, 
            epochs=2000
        )
        C_pred_val = C_MixG_MultiSigma(X_val, r_full_train, tau_full_train, trained_params["sigmas"], trained_params["mus"], trained_params["pis"])
        mse_val = loss(C_pred_val, C_val)
        print(f"---> Validation MSE for weight={weight:.6f}: {mse_val:.4f}")
        
        return mse_val

    # --- 3. 运行贝叶斯优化 ---
    print("\n--- 3. Running Bayesian Optimization to find best smoothness_weight... ---")
    res_gp = gp_minimize(objective, space, n_calls=10, random_state=42) # n_calls可以根据需要调整
    best_weight = res_gp.x[0]
    print(f"\n--- Optimization finished. Best smoothness_weight found: {best_weight:.6f} ---")

    # --- 4. 使用最优权重进行最终训练 ---
    print("\n--- 4. Training final model on FULL training dataset... ---")
    final_params = optimize_hybrid_model(
        C_obs=C_obs_full_train, n=len(X_full_train), X=X_full_train, S_t=S_t_full_train,
        r=r_full_train, tau=tau_full_train, d=0, 
        sigma_best_anchor=0.0647,
        smoothness_weight=best_weight, 
        epochs=5000
    )
    mu_final, pi_final, sigmas_final = final_params["mus"], final_params["pis"], final_params["sigmas"]

    # --- 5. 最终评估与可视化 ---
    print("\n--- 5. Final Evaluation and Visualization ---")
    # 训练集拟合
    C_fitted = C_MixG_MultiSigma(X_full_train, r_full_train, tau_full_train, sigmas_final, mu_final, pi_final)
    fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
    ax_fit.scatter(X_full_train, C_obs_full_train, label='Observed C (Full Train)')
    ax_fit.plot(X_full_train, C_fitted, 'r-o', markersize=4, label=f'Fitted C (best_weight={best_weight:.4f})')
    ax_fit.set_title('Final Fit on Full Training Data')
    ax_fit.legend(); ax_fit.grid(True)
    plt.show()

    # 测试集预测
    X_test = test_data['strick_price(X)'].values
    C_obs_test = test_data['option_price(C_bid)'].values
    tau_test = test_data['tau(T/365)'].iloc[0]
    r_test = test_data['r'].iloc[0]
    C_pred_test = C_MixG_MultiSigma(X_test, r_test, tau_test, sigmas_final, mu_final, pi_final)
    test_mse = loss(C_pred_test, C_obs_test)
    print(f"Final Mean Squared Error (MSE) on Test Set: {test_mse:.4f}")

    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    ax_pred.scatter(X_test, C_obs_test, label='Observed C (Test)')
    ax_pred.plot(X_test, C_pred_test, 'g-o', markersize=4, label='Predicted C (Test)')
    ax_pred.set_title('Prediction on Test Data with Final Model')
    ax_pred.legend(); ax_pred.grid(True)
    plt.show()
    
    # SPD 绘图
    print("\n--- 6. Calculating and plotting State-Price Density (SPD)... ---")
    fig_spd = None
    try:
        import torch
        from modules.BS_MixG_model import C_MixG_Torch_MultiSigma
        mu_opt_torch = torch.tensor(mu_final, dtype=torch.float32)
        pi_opt_torch = torch.tensor(pi_final, dtype=torch.float32)
        sigmas_opt_torch = torch.tensor(sigmas_final, dtype=torch.float32)
        r_torch = torch.tensor(r_full_train, dtype=torch.float32)
        tau_torch = torch.tensor(tau_full_train, dtype=torch.float32)
        X_dense = torch.linspace(X_full_train.min() * 0.8, X_full_train.max() * 1.2, 200, requires_grad=True)
        C_dense = C_MixG_Torch_MultiSigma(X_dense, r_torch, tau_torch, sigmas_opt_torch, mu_opt_torch, pi_opt_torch)
        first_derivative = torch.autograd.grad(C_dense.sum(), X_dense, create_graph=True)[0]
        second_derivative = torch.autograd.grad(first_derivative.sum(), X_dense)[0]
        spd = torch.exp(r_torch * tau_torch) * second_derivative
        
        fig_spd, ax_spd = plt.subplots(figsize=(10, 6))
        ax_spd.plot(X_dense.detach().numpy(), spd.detach().numpy(), label='Estimated SPD (Hybrid Model)')
        ax_spd.set_title('State-Price Density from Hybrid Model')
        ax_spd.set_xlabel('Future Asset Price (S_T)')
        ax_spd.set_ylabel('Density')
        ax_spd.set_ylim(bottom=0, top=np.percentile(spd.detach().numpy(), 99.5))
        ax_spd.legend(); ax_spd.grid(True)
        plt.show()
    except Exception as e:
        print(f"An error occurred during SPD plot generation: {e}")

    # --- 7. 保存结果 ---
    print("\n--- 7. Saving results to file... ---")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("results_bayesopt", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 保存参数
    np.savez(os.path.join(output_dir, "optimized_parameters.npz"), 
             mus=mu_final, 
             pis=pi_final, 
             sigmas=sigmas_final,
             best_smoothness_weight=best_weight)

    # 保存结果摘要
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"--- Summary of Bayesian Optimized Hybrid Model Run on {timestamp} ---\n")
        f.write(f"Best Smoothness Weight Found: {best_weight:.6f}\n")
        f.write(f"Final Test Set MSE: {test_mse:.6f}\n\n")
        f.write("--- Final Optimized Parameters (Sorted by Mu) ---\n")
        f.write("Mus:\n" + np.array2string(mu_final, precision=4) + "\n\n")
        f.write("Pis:\n" + np.array2string(pi_final, precision=4) + "\n\n")
        f.write("Sigmas:\n" + np.array2string(sigmas_final, precision=4) + "\n")

    # 保存图像
    fig_fit.savefig(os.path.join(output_dir, "training_fit_plot.png"))
    fig_pred.savefig(os.path.join(output_dir, "test_prediction_plot.png"))
    if fig_spd:
        fig_spd.savefig(os.path.join(output_dir, "spd_plot.png"))
    
    print("All results have been saved successfully.")

if __name__ == "__main__":
    main()