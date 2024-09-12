import numpy as np
from modules.BS_Theoretical_Model import BS_Theoretical_Value, add_noise_to_option_values
from matplotlib import pyplot as plt
from optimize import simulation_LOOCV
import warnings

warnings.filterwarnings('ignore')

# np.random.seed(0)

n = 25
# 从1000-1700均匀取25个数
X = np.linspace(1000, 1700, n)
T = 30
r = 0.045
d = 0.025
ST = 1365
tau = T / 365
sigma_1000 = 0.2  # 当 X = 1000 时的波动率
sigma_1700 = 0.4  # 当 X = 1700 时的波动率


if __name__ == '__main__':
    BS_Value = BS_Theoretical_Value(X, ST, T, r, d, sigma_1000, sigma_1700)
    C_obs = add_noise_to_option_values(BS_Value, X)
    print(type(C_obs), type(X))
    C_pred, C_pred_opt_1, C_pred_opt_2, loss_opt = simulation_LOOCV(0.071, C_obs, n, X, r, tau, d)
    # 绘制图像
    plt.plot(X, C_obs, label='Observed')
    plt.plot(X, C_pred_opt_2, label='Optimized')
    plt.legend()
    plt.show()