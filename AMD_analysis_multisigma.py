# amd_analysis_multisigma.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 从新模块导入
from optimize_multisigma import optimize_end_to_end
from modules.BS_MixG_model import C_MixG_MultiSigma 
from utils import loss

def main():
    # --- 1. 数据加载与处理 (与之前相同) ---
    print("--- 1. Loading and processing AMD option data... ---")
    path = 'data/real_2018_6_7_AMD_data.csv'
    data = pd.read_csv(path)
    tau_values = np.sort(np.unique(data['tau(T/365)']))
    train_data = data[data['tau(T/365)'] == tau_values[0]].copy()
    test_data = data[data['tau(T/365)'] == tau_values[1]].copy()
    train_data['r'] = train_data['r'] / 100
    test_data['r'] = test_data['r'] / 100
    print("Data loaded and split.\n")

    # --- 2. 准备训练变量 (与之前相同) ---
    print("--- 2. Preparing variables for model training... ---")
    X_train = train_data['strick_price(X)'].values
    C_obs_train = train_data['option_price(C_bid)'].values
    n_train = len(C_obs_train)
    r_train = train_data['r'].iloc[0]
    tau_train = train_data['tau(T/365)'].iloc[0]
    S_t_train = train_data['close'].iloc[0]

    # --- 3. 执行端到端优化 ---
    optimized_params = optimize_end_to_end(
        C_obs=C_obs_train,
        n=n_train,
        X=X_train,
        S_t=S_t_train,
        r=r_train,
        tau=tau_train,
        d=0
    )
    mu_final = optimized_params["mus"]
    pi_final = optimized_params["pis"]
    sigmas_final = optimized_params["sigmas"]
    
    print("\nFinal Optimized Sigmas:")
    print(np.round(sigmas_final, 4))

    # --- 4. 可视化训练集拟合结果 ---
    C_fitted = C_MixG_MultiSigma(X_train, r_train, tau_train, sigmas_final, mu_final, pi_final)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, C_obs_train, label='Observed C (Train)')
    plt.plot(X_train, C_fitted, 'r-o', markersize=4, label=f'Fitted C (Multi-Sigma Model)')
    plt.title('Final Fit on AMD Training Data (Multi-Sigma Model)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 5. 在测试集上进行预测 ---
    X_test = test_data['strick_price(X)'].values
    C_obs_test = test_data['option_price(C_bid)'].values
    tau_test = test_data['tau(T/365)'].iloc[0]
    r_test = test_data['r'].iloc[0]
    
    C_pred_test = C_MixG_MultiSigma(X_test, r_test, tau_test, sigmas_final, mu_final, pi_final)
    test_mse = loss(C_pred_test, C_obs_test)
    print(f"\nMean Squared Error (MSE) on Test Set: {test_mse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, C_obs_test, label='Observed C (Test)')
    plt.plot(X_test, C_pred_test, 'g-o', markersize=4, label='Predicted C (Test)')
    plt.title('Prediction on AMD Test Data (Multi-Sigma Model)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 6. 计算并绘制状态价格密度 (SPD) ---
    print("\n--- 6. Calculating and plotting State-Price Density (SPD) for Multi-Sigma Model... ---")
    try:
        import torch
        # 确保导入的是处理多元sigma的PyTorch模型函数
        from modules.BS_MixG_model import C_MixG_Torch_MultiSigma

        # 将最终优化出的参数转换为PyTorch张量
        mu_opt_torch = torch.tensor(mu_final, dtype=torch.float32)
        pi_opt_torch = torch.tensor(pi_final, dtype=torch.float32)
        sigmas_opt_torch = torch.tensor(sigmas_final, dtype=torch.float32) # 使用优化出的sigma向量
        r_torch = torch.tensor(r_train, dtype=torch.float32)
        tau_torch = torch.tensor(tau_train, dtype=torch.float32)

        # 为绘制平滑曲线，创建一个密集的执行价格范围
        X_dense = torch.linspace(X_train.min() * 0.8, X_train.max() * 1.2, 200, requires_grad=True)

        # 使用多元sigma模型计算期权价格
        C_dense = C_MixG_Torch_MultiSigma(X_dense, r_torch, tau_torch, sigmas_opt_torch, mu_opt_torch, pi_opt_torch)
        
        # 利用PyTorch的自动微分计算一阶导数 (dC/dX)
        # create_graph=True 允许我们进行二次求导
        first_derivative = torch.autograd.grad(C_dense.sum(), X_dense, create_graph=True)[0]

        # 计算二阶导数 (d^2C/dX^2)
        second_derivative = torch.autograd.grad(first_derivative.sum(), X_dense)[0]

        # 根据公式计算SPD: f(S_T) = exp(r*tau) * (d^2C/dX^2)
        spd = torch.exp(r_torch * tau_torch) * second_derivative

        # --- 绘图 ---
        plt.figure(figsize=(10, 6))
        plt.plot(X_dense.detach().numpy(), spd.detach().numpy(), label='Estimated SPD (Multi-Sigma)')
        plt.title('State-Price Density from Multi-Sigma Model')
        plt.xlabel('Future Asset Price (S_T)')
        plt.ylabel('Density')
        # 设定y轴范围，防止极端值影响图像可读性
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