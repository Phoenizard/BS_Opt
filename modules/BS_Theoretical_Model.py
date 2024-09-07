import numpy as np
from scipy.stats import norm

def get_sigma(S, S_low, sigma_low, S_high, sigma_high):
    return sigma_low + (sigma_high - sigma_low) * (S - S_low) / (S_high - S_low)

def BS_Theoretical_Value(ST, T, r, d, sigma, X):
    """
    ST: Current price of the underlying asset, 
    T: Time to expiration (in days)
    r: Risk-free rate
    d: Dividend yield
    sigma: Volatility 波动率
    X: Strike price
    Calculate the theoretical value of a call option
    """
    # Ensure X is a numpy array for vectorized operations
    X = np.array(X)

    # 生成波动率序列
    sigma_set = [get_sigma(S, 1000, 0.2, 1700, 0.4) for S in X]

    sigma = np.array(sigma_set)
    tau = T / 365
    d1 = (np.log(ST / X) + (r - d + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    OptionValue = np.exp(-d * tau) * ST * norm.cdf(d1) - np.exp(-r * tau) * X * norm.cdf(d2)
    return OptionValue

if __name__ == '__main__':
    sigma_1000 = 0.2  # 当 X = 1000 时的波动率
    sigma_1700 = 0.4  # 当 X = 1700 时的波动率
    # Test the function
    ST = 1365
    T = 30
    r = 0.045
    d = 0.025
    sigma = 0.2
    X = np.linspace(1000, 1700, 25)
    C_theoretical = BS_Theoretical_Value(ST, T, r, d, sigma, X)
    print(C_theoretical)