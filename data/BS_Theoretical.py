import numpy as np
from scipy.stats import norm

# Theoretical values for the Black-Scholes model
def BS_Theoretical_Value(ST: float, T: int, r: float, d: float, sigma: float, X: float):
    # ST: Current price of the underlying asset, 
    # T: Time to expiration (in days)
    # r: Risk-free rate
    # d: Dividend yield
    # sigma: Volatility 波动率
    # X: Strike price
    # Calculate the theoretical value of a call option
    tau = T / 365
    d1 = (np.log(ST / X) + (r - d + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    OptionValue = np.exp(-d * tau) * ST * norm.cdf(d1) - np.exp(-r * tau) * X * norm.cdf(d2)
    return OptionValue