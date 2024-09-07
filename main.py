import numpy as np
from modules.BS_MixG_model import C_MixG
from modules.BS_Theoretical_Model import BS_Theoretical_Value, add_noise_to_option_values
from matplotlib import pyplot as plt
from scipy.optimize import minimize, LinearConstraint

np.random.seed(0)

n = 25
# 从1000-1700均匀取25个数
X = np.linspace(1000, 1700, n)
T = 30
r = 0.045
d = 0.025
ST = 1365
tau = T / 365
sigma_0 = 0.2
sigma_1000 = 0.2  # 当 X = 1000 时的波动率
sigma_1700 = 0.4  # 当 X = 1700 时的波动率

def loss(C_pred, C_true):
    if C_pred.shape != C_true.shape:
        raise ValueError("C_pred and C_true should have the same shape")
    return np.mean((C_pred - C_true) ** 2)

# 计算Expection价值
def Expection(mu, pi):
    return np.dot(np.exp(mu), pi) * np.exp(sigma_0 ** 2 / 2)

# 计算看涨期权下未来价值
def FutureValue(r, d, tau, X):
    return (np.exp((r - d) * tau) * X).mean()

# 目标函数
def objective(pi, X, r, tau, sigma_0, mu, C_obs):
    # C_pred 通过 C_MixG 计算
    C_pred = C_MixG(X, r, tau, sigma_0, mu, pi)
    return loss(C_pred, C_obs)

def constraint_expection(pi, mu):
    # 计算期望值和未来价值
    E_value = Expection(mu, pi)
    return E_value - FV

def Quad_Optimize(X, r, tau, sigma_0, mu, pi_init, C_obs):
    expection_constraint = {'type': 'eq', 'fun': constraint_expection, 'args': (mu,)}
    bounds = [(0, None)] * (n + 1) # pi之和为1
    linear_constraint = LinearConstraint(np.ones(n + 1), 1, 1)
    res = minimize(objective, x0=pi_init, args=(X, r, tau, sigma_0, mu, C_obs), method='SLSQP', 
                constraints=[expection_constraint, linear_constraint], bounds=bounds,
                options={'disp': True, 'ftol': 1e-9})
    return res

if __name__ == '__main__':
    BS_Value = BS_Theoretical_Value(X, ST, T, r, d, sigma_1000, sigma_1700)
    C_obs = add_noise_to_option_values(BS_Value, X)
    # 优化参数
    mu = np.random.uniform(low=7.107, high=7.265, size=(n + 1))
    pi_init = np.ones(n + 1) / (n + 1)  # 初始猜测的 pi
    C_pred = C_MixG(X, r, tau, sigma_0, mu, pi_init)
    # 损失函数
    FV = FutureValue(r, d, tau, X)
    res = Quad_Optimize(X, r, tau, sigma_0, mu, pi_init, C_obs)
    pi_opt = res.x
    C_opt_pi = C_MixG(X, r, tau, sigma_0, mu, pi_opt)
    print("Optimized pi: ", pi_opt)
    print("Optimized loss: ", res.fun)
    print("Optimized Expection: ", Expection(mu, pi_opt), "Future Value: ", FV)
    # 画图
    plt.plot(X, C_obs, label='Observed')
    plt.plot(X, C_pred, label='Predicted')
    plt.plot(X, C_opt_pi, label='Optimized')
    plt.xlabel('Strike Price (X)')
    plt.ylabel('Option Price (C)')
    plt.legend()
    plt.show()
