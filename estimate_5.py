import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
# 禁用 cvxopt 输出
solvers.options['show_progress'] = False

def european_option_price(X, r, tau, mu, sigma, weights, num_intervals=1000):
    # 定义积分区间和步长
    ST_values = np.linspace(0, 8 * X, num_intervals)
    delta_ST = ST_values[1] - ST_values[0]

    # 计算每个 ST 点上的概率密度函数值和对应的收益
    pdf_values = sum([w * lognorm.pdf(ST_values, s, scale=np.exp(m))
                      for w, m, s in zip(weights, mu, sigma)])
    payoffs = np.maximum(ST_values - X, 0)

    # 进行黎曼积分并考虑贴现因子
    option_price = np.exp(-r * tau) * np.sum(payoffs * pdf_values * delta_ST)
    
    return option_price


# 二次规划更新权重
def update_weights_via_quadratic_program(mu, sigma, weights, X, C_observed, r, tau, St):
    n = len(mu)  # 混合权重的数量
    
    C_predicted = np.array([european_option_price(x, r, tau, mu, sigma, weights) for x in X])

    Q = np.eye(n) * (sigma**2)  # 在 sigma 相同的情况下，Q 矩阵可以简化
    Q = matrix(Q, tc='d')
    q = matrix((C_observed - C_predicted), tc='d')
    G = matrix(-np.eye(n), tc='d')
    h = matrix(np.zeros(n), tc='d')


    # Ef = np.sum(np.exp(mu + sigma**2 / 2) * weights)
    F = St * np.exp(np.mean(sigma**2) / 2) * np.sum(weights * np.exp(mu))

    # 等式约束：权重总和为 1，以及两个公式相等的约束
    A = matrix(np.vstack([np.ones(n), np.exp(mu + sigma**2 / 2)]), tc='d')
    b = matrix([1.0, float(F)], tc='d')

    sol = solvers.qp(Q, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    

    return weights

# 牛顿迭代更新均值
def update_means_via_newton_raphson(mu, sigma, weights, X, C_observed, r, tau, max_iter=10, tol=1e-4):
    for _ in range(max_iter):
    # delta_norm = np.inf
    # _ = 0
    # while delta_norm > tol:
        grad = np.zeros_like(mu)
        hessian = np.zeros((len(mu), len(mu)))

        for i in range(len(X)):
            C_pred = european_option_price(X[i], r, tau, mu, sigma, weights)
            diff = C_pred - C_observed[i]

            for j in range(len(mu)):
                dC_dmu =  derivative_of_european_option_price_wrt_mu(X[i], r, tau, mu, sigma, weights, j)
                d2C_dmu2 = second_derivative_of_european_option_price_wrt_mu(X[i], r, tau, mu, sigma, weights, j)
                
                grad[j] += 2 * diff * dC_dmu
                hessian[j, j] += 2 * (dC_dmu**2 + diff * d2C_dmu2)

        try:
            delta = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian + np.eye(len(mu)) * 1e-6, -grad, rcond=None)[0]

        mu = mu + delta
        delta_norm = np.linalg.norm(delta)
        if np.linalg.norm(delta) < tol:
            break

        # _ += 1
        # print(f"Newton iteration: {_}", f"Delta norm: {delta_norm}")
    return mu


def derivative_of_european_option_price_wrt_mu(X, r, tau, mu, sigma, weights, j):
    ST_values = np.linspace(0.001, 8 * X, 1000)
    dC_dmu = weights[j] * np.trapz(
        np.exp(-r * tau) * lognorm.pdf(ST_values, sigma[j], scale=np.exp(mu[j])) * (ST_values - X) * (np.log(ST_values) - mu[j]) / sigma[j]**2,
        ST_values
    )
    return dC_dmu


def second_derivative_of_european_option_price_wrt_mu(X, r, tau, mu, sigma, weights, j):
    ST_values = np.linspace(0.001, 8 * X, 1000)
    d2C_dmu2 = weights[j] *  np.trapz(
        np.exp(-r * tau) * lognorm.pdf(ST_values, sigma[j], scale=np.exp(mu[j])) * (ST_values - X) * ((np.log(ST_values) - mu[j])**2 / sigma[j]**4 - 1/sigma[j]**2),
        ST_values
    )
    return d2C_dmu2


def weighted_least_squares_error(X, C_observed, r, tau, mu, sigma, weights):
    C_pred = np.array([european_option_price(x, r, tau, mu, sigma, weights) for x in X])
    error = np.sum((C_pred - C_observed) ** 2)
    return error


def sigema_estimation_via_loocv(X, C_observed, r, tau, mu, weights, sigma,St):
    best_sigma = sigma
    best_loocv_error = float('inf')
    sigma_range = np.linspace(2, 3.5, 5)  # 选择一个合理的 sigma 范围

    best_mu = None
    best_weights = None

    for sigma_candidate in sigma_range:
        counter = 0

        sigma_trial = np.full_like(sigma, sigma_candidate)
        
        # 初始化候选参数，并清空以确保不累积
        mu_candidate = []
        weights_candidate = []
        current_loocv_error = 0
        
        for i in range(len(X)):
            x_train = np.delete(X, i)
            C_observed_train = np.delete(C_observed, i)
                                        
            updated_weight = update_weights_via_quadratic_program(mu, sigma_trial, weights, x_train, C_observed_train, r, tau,St)
            updated_mu = update_means_via_newton_raphson(mu, sigma_trial, updated_weight, x_train, C_observed_train, r, tau)
            
            mu_candidate.append(updated_mu)
            weights_candidate.append(updated_weight)
            
            C_test_pred = european_option_price(X[i], r, tau, updated_mu, sigma_trial, updated_weight)
            current_loocv_error += (C_test_pred - C_observed[i]) ** 2
            print(f"{i}th loops in sigma:{sigma_candidate}")
        # 计算LOOCV平均误差
        current_loocv_error /= len(X)

        counter += 1
        print(f"test the {counter}th sigma is:{sigma_candidate}",f"current_loocv_error: {current_loocv_error}")


        # 如果当前 sigma_candidate 更优，则更新最佳参数
        if current_loocv_error < best_loocv_error:
            best_loocv_error = current_loocv_error
            best_sigma = sigma_trial
            best_mu_candidate = mu_candidate.copy()  
            best_weights_candidate = weights_candidate.copy()

    error = np.inf
    for i in range(len(X)-1):

        if weighted_least_squares_error(X, C_observed, r, tau, best_mu_candidate[i], best_sigma, best_weights_candidate[i]) < error:
            best_mu = best_mu_candidate[i]
            best_weights = best_weights_candidate[i]
            error = weighted_least_squares_error(X, C_observed, r, tau, best_mu_candidate[i], best_sigma, best_weights_candidate[i])
    
    
    return best_sigma, best_mu, best_weights



##--------------------------------------------##


# 设置随机种子以确保可重复性
np.random.seed(42)

# 基本参数
St = 1365  # 当前指数价格
r = 0.045  # 短期利率
q = 0.025  # 股息收益
tau = 30 / 365  # 到期时间 30 天

# 生成执行价格 X 均匀分布在 1000 和 1700 之间，数量为 25
n_samples = 25
X = np.linspace(1000, 1700, n_samples)

# 计算波动率 sigma 随执行价格线性变化
sigma_low = 0.40
sigma_high = 0.20
sigma = sigma_low + (X - 1000) / (1700 - 1000) * (sigma_high - sigma_low)

# 计算无噪声的理论期权价格
def black_scholes_call(St, X, tau, r, sigma, q=0.025):
    d1 = (np.log(St / X) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    call_price = np.exp(-q * tau) * St * norm.cdf(d1) - np.exp(-r * tau) * X * norm.cdf(d2)
    return call_price

C_theoretical = black_scholes_call(St, X, tau, r, sigma, q)

# 添加噪声，噪声范围从 3% 到 18%
noise = np.linspace(0.03, 0.18, n_samples) * C_theoretical
C_observed = C_theoretical + np.random.uniform(-1, 1, n_samples) * noise

# 初始猜测
n = len(X) - 1  # 假设 n+1 个对数正态分布

mu_initial = np.linspace(-300, 300, n)
sigma_initial = np.array([1.3] * n)
weights_initial = np.full(n, 1/n)

# 估计状态价格密度
sigma_estimated, mu_estimated, mixture_weights_estimated = sigema_estimation_via_loocv(
    X, C_observed, r, tau, mu_initial, weights_initial,sigma_initial,  St)

# 打印结果
print("估计的mu:", mu_estimated)
print("估计的sigma:", sigma_estimated)
print("估计的混合权重:", mixture_weights_estimated)

# 绘图函数
def plot_option_prices(X, r, tau, mu, sigma, weights):
    prices = np.linspace(min(X) / 2, max(X) * 2, 400)
    option_prices = [european_option_price(x, r, tau, mu, sigma, weights) for x in prices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices, option_prices, label='Option Price')
    plt.scatter(X, C_observed, color='red', marker='x', label='Observed Prices')
    plt.title('European Call Option Price by Asset Price')
    plt.xlabel('Asset Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用估计参数绘图
plot_option_prices(X, r, tau, mu_estimated, sigma_estimated, mixture_weights_estimated)