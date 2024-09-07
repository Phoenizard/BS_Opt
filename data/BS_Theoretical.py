import numpy as np
from scipy.stats import norm

# Black-Scholes 模型的理论期权价值
def BS_Theoretical_Value(ST: float, T: int, r: float, d: float, sigma: float, X: float):
    # ST: 当前标的资产价格
    # T: 距离到期日的天数
    # r: 无风险利率
    # d: 股息率
    # sigma: 波动率
    # X: 行权价格
    # 计算欧式看涨期权的理论价值
    tau = T / 365
    d1 = (np.log(ST / X) + (r - d + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    OptionValue = np.exp(-d * tau) * ST * norm.cdf(d1) - np.exp(-r * tau) * X * norm.cdf(d2)
    return OptionValue

# 噪声添加函数
def add_noise_to_option_values(ST, T, r, d, sigma, X, low_noise=0.03, high_noise=0.18):
    # 计算理论期权价值
    C_theoretical = np.array([BS_Theoretical_Value(ST, T, r, d, sigma, x) for x in X])

    # 噪声比例：从价内3%线性变化到价外18%
    noise_percentages = np.linspace(low_noise, high_noise, len(X))

    # 生成随机噪声，按噪声比例生成对应的随机值
    np.random.seed(42)  # 保持一致性
    noise = noise_percentages * C_theoretical * np.random.uniform(-1, 1, size=len(X))

    # 将噪声添加到理论期权价格中
    C_noisy = C_theoretical + noise

    return C_theoretical, C_noisy

if __name__ == '__main__':
    # 测试参数
    ST = 1365  # 标的资产价格
    T = 30  # 距离到期天数
    r = 0.045  # 无风险利率
    d = 0.025  # 股息率
    sigma = 0.2  # 波动率
    X = np.linspace(1000, 1700, 25)  # 行权价格范围

    # 计算带有噪声的期权价格
    C_theoretical, C_noisy = add_noise_to_option_values(ST, T, r, d, sigma, X)

    # 打印结果
    print("理论期权价格：", C_theoretical)
    print("添加噪声后的期权价格：", C_noisy)