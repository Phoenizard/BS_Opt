import numpy as np
from scipy.stats import norm

# 根据行权价格动态计算波动率
def get_sigma(S, S_low, sigma_low, S_high, sigma_high):
    """
    根据 S（行权价格）动态生成对应的波动率。
    当 S = S_low 时，波动率为 sigma_low；
    当 S = S_high 时，波动率为 sigma_high；
    其余 S 值根据线性插值计算。
    """
    return sigma_low + (sigma_high - sigma_low) * (S - S_low) / (S_high - S_low)

# Black-Scholes 模型计算期权理论价格
def BS_Theoretical_Value(X, ST, T, r, d, sigma_low=0.2, sigma_high=0.4):
    """
    ST: 当前标的资产价格
    T: 距离到期天数
    r: 无风险利率
    d: 股息率
    X: 行权价格数组
    sigma_low: 当行权价格低（如 1000）时的波动率
    sigma_high: 当行权价格高（如 1700）时的波动率
    """
    # 确保 X 是 NumPy 数组以实现向量化操作
    X = np.array(X)

    # 生成波动率序列
    sigma_set = get_sigma(X, 1000, sigma_low, 1700, sigma_high)
    
    # 转换为 NumPy 数组
    sigma = np.array(sigma_set)
    tau = T / 365  # 到期时间（年）

    # 计算 d1 和 d2
    d1 = (np.log(ST / X) + (r - d + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # 计算期权价值
    OptionValue = np.exp(-d * tau) * ST * norm.cdf(d1) - np.exp(-r * tau) * X * norm.cdf(d2)
    return OptionValue

# 添加噪声函数
def add_noise_to_option_values(C_theoretical, X, low_noise=0.03, high_noise=0.18):
    """
    为理论期权价格添加噪声，噪声从3%到18%线性增长
    C_theoretical: 理论期权价格数组
    X: 行权价格数组
    low_noise: 噪声的最小比例（价内期权）
    high_noise: 噪声的最大比例（价外期权）
    """
    # 噪声比例：从价内3%线性变化到价外18%
    noise_percentages = np.linspace(low_noise, high_noise, len(X))

    # 生成随机噪声，按噪声比例生成对应的随机值
    # np.random.seed(42)  # 保持一致性
    noise = noise_percentages * C_theoretical * np.random.uniform(-1, 1, size=len(X))

    # 将噪声添加到理论期权价格中
    C_noisy = C_theoretical + noise

    return C_noisy

if __name__ == '__main__':
    sigma_1000 = 0.2  # 当 X = 1000 时的波动率
    sigma_1700 = 0.4  # 当 X = 1700 时的波动率

    # 测试参数
    ST = 1365  # 当前标的资产价格
    T = 30  # 距离到期天数
    r = 0.045  # 无风险利率
    d = 0.025  # 股息率
    X = np.linspace(1000, 1700, 25)  # 行权价格范围

    # 计算期权理论价格
    C_theoretical = BS_Theoretical_Value(X, ST, T, r, d, sigma_1000, sigma_1700)

    # 为理论期权价格添加噪声
    C_noisy = add_noise_to_option_values(C_theoretical, X)

    # 打印结果
    print("理论期权价格：", C_theoretical)
    print("添加噪声后的期权价格：", C_noisy)