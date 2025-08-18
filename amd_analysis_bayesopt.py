# amd_analysis_bayesopt.py

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from skopt import gp_minimize # 从 scikit-optimize 导入贝叶斯优化器
from skopt.space import Real # 定义搜索空间
from skopt.utils import use_named_args

# 导入您项目中的函数
from optimize_hybrid import optimize_hybrid_model
from modules.BS_MixG_model import C_MixG_MultiSigma
from utils import loss

# --- 初始设置 ---
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# --- 1. 数据加载与处理 ---
print("--- 1. Loading data... ---")
# ... (这部分与之前完全相同，此处省略)
path = 'data/real_2018_6_7_AMD_data.csv'
data = pd.read_csv(path)
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
X_train_sub, X_val, C_train_sub, C_val = train_test_split(
    X_full_train, C_obs_full_train, test_size=0.2, random_state=42
)
n_train_sub = len(X_train_sub)


# --- 2. 定义贝叶斯优化的搜索空间和目标函数 ---
# 我们要搜索的参数是 smoothness_weight，范围是 1e-5 到 1.0，使用对数均匀分布
space  = [Real(1e-5, 1.0, "log-uniform", name='smoothness_weight')]

# 这是贝叶斯优化器要最小化的目标函数
@use_named_args(space)
def objective(**params):
    weight = params['smoothness_weight']
    print(f"\n---> Testing smoothness_weight = {weight:.6f}")

    # a. 在训练子集上训练
    trained_params = optimize_hybrid_model(
        C_obs=C_train_sub, n=n_train_sub, X=X_train_sub, S_t=S_t_full_train,
        r=r_full_train, tau=tau_full_train, d=0, 
        sigma_best_anchor=0.0647, # 使用我们已知的最佳锚点
        smoothness_weight=weight, 
        epochs=2000 # 在搜索时可以适当减少epoch以加快速度
    )
    # b. 在验证集上评估
    C_pred_val = C_MixG_MultiSigma(X_val, r_full_train, tau_full_train, trained_params["sigmas"], trained_params["mus"], trained_params["pis"])
    mse_val = loss(C_pred_val, C_val)
    print(f"---> Validation MSE for weight={weight:.6f}: {mse_val:.4f}")
    
    return mse_val

# --- 3. 运行贝叶斯优化 ---
print("\n--- 3. Running Bayesian Optimization to find best smoothness_weight... ---")
# n_calls 是总的尝试次数
res_gp = gp_minimize(objective, space, n_calls=20, random_state=42)

best_weight = res_gp.x[0]
print(f"\n--- Optimization finished. Best smoothness_weight found: {best_weight:.6f} ---")

# --- 4. 使用最优权重进行最终训练 ---
print("\n--- 4. Training final model on FULL training dataset... ---")
final_params = optimize_hybrid_model(
    C_obs=C_obs_full_train, n=len(X_full_train), X=X_full_train, S_t=S_t_full_train,
    r=r_full_train, tau=tau_full_train, d=0, 
    sigma_best_anchor=0.0647,
    smoothness_weight=best_weight, 
    epochs=5000 # 最终训练使用更多epoch
)

# --- 5. 最终评估与可视化 ---
# ... (这部分与之前的脚本完全相同，用于评估和绘图)