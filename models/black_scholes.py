import numpy as np
from scipy.stats import norm
from .base import OptionPricingModel

class BlackScholes(OptionPricingModel):
    def __init__(self, S, K, T, r, sigma):
        """
        S = Spot price
        K = Strike price
        T = Time to maturity (years)
        r = Risk-free interest rate
        sigma = Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def _calculate_d1_d2(self):
        """Helper function to calculate d1 and d2 components."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def calculate_price(self, option_type='call'):
        """Calculates the fair price of a European option."""
        d1, d2 = self._calculate_d1_d2()
        if option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return None

    def calculate_greeks(self, option_type='call'):
        """Calculates Delta, Gamma, Theta, Vega, and Rho."""
        d1, d2 = self._calculate_d1_d2()
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)

        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) 
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else: # put
            delta = norm.cdf(d1) - 1
            theta = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) 
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}