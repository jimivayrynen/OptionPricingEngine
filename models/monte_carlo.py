import numpy as np
from .base import OptionPricingModel

class MonteCarloPricing(OptionPricingModel):
    def __init__(self, S, K, T, r, sigma, iterations=10000):
        """
        S = Spot price
        K = Strike price
        T = Time to maturity (years)
        r = Risk-free rate
        sigma = Volatility
        iterations = Number of simulated paths
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.iterations = iterations

    def calculate_price(self, option_type='call'):
        """
        Calculates price by simulating asset prices using Geometric Brownian Motion.
        """
        if self.T <= 0:
            return 0.0

        # 1. Generate random shocks (Z) from standard normal distribution
        Z = np.random.standard_normal(self.iterations)

        # 2. Calculate asset price at maturity (ST)
        drift = (self.r - 0.5 * self.sigma ** 2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        
        ST = self.S * np.exp(drift + diffusion)

        # 3. Calculate Payoff
        if option_type == 'call':
            payoff = np.maximum(ST - self.K, 0)
        elif option_type == 'put':
            payoff = np.maximum(self.K - ST, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        # 4. Discount back to present value
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        
        return price