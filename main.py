import numpy as np
from modules.BS_Theoretical_Model import BS_Theoretical_Value, add_noise_to_option_values
from matplotlib import pyplot as plt
from optimize import simulation_LOOCV
from optimize import calculate_and_plot_expression

import warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')

# np.random.seed(0)

n = 25
# 从1000-1700均匀取25个数
X = np.linspace(1000, 1700, n)
T = 30
r = 1.045
d = 0.025
ST = 1365
tau = T / 365
sigma_1000 = 0.2  # 当 X = 1000 时的波动率
sigma_1700 = 0.4  # 当 X = 1700 时的波动率


if __name__ == '__main__':
    BS_Value = BS_Theoretical_Value(X, ST, T, r, d, sigma_1000, sigma_1700)
    C_obs = add_noise_to_option_values(BS_Value, X)
    print(type(C_obs), type(X))
    C_pred, C_pred_opt_1, C_pred_opt_2, loss_opt,mu_opt, pi_opt = simulation_LOOCV(0.071, C_obs, n, X, r, tau, d)

    print('Optimized mu:', mu_opt,"type:",type(mu_opt))
    print('Optimized pi:', pi_opt,"type:",type(pi_opt))
    sigma = 0.071
    calculate_and_plot_expression(pi_opt, mu_opt, sigma, 1500, 1.045, tau, 1.5*ST)


    plt.plot(X, C_obs, label='Observed')
    plt.plot(X, C_pred_opt_2, label='Optimized')
    plt.legend()
    plt.show()