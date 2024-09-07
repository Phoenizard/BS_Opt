import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
def F(X):
    return 1 - norm.cdf(X)

def C_MixG(X, r, tau, sigma, mu, pi) -> np.ndarray:
    """
    r: Risk-free rate
    d: Dividend yield
    tau: Time to expiration (in years)
    sigma: Volatility
    mu: Mean of the lognormal distribution
    X: Strike prices (can be a vector)
    Calculate the theoretical value of call options in G Mixtures model
    """

    # Ensure X is a numpy array for vectorized operations
    X = np.array(X)
    
    # Calculate d1 and d2 as vectors
    d1 = np.exp(-r * tau + mu + 0.5 * sigma ** 2)[:, np.newaxis] * F((np.log(X) - (mu[:, np.newaxis] + sigma ** 2)) / sigma)
    d2 = np.exp(-r * tau) * X * F((np.log(X) - mu[:, np.newaxis]) / sigma)
    
    # Sum over the mixture components
    C = np.sum((d1 - d2) * pi[:, np.newaxis], axis=0)
    
    return C


if __name__ == '__main__':
    # Test the function
    r = 0.045
    d = 0.025
    tau = 30 / 365
    sigma = 0.2
    mu_init = np.random.uniform(low=7.107, high=7.265, size=26)
    # Convert to numpy array
    pi_init = np.ones(26) / 26
    X = np.linspace(1000, 1700, 25)
    C_pred = (C_MixG(X, r, tau, sigma, mu_init, pi_init))
