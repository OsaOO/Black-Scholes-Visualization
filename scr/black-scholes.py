"""
Author: OsaO
Last Updated By: OsaO
Last Updated Date: 03/02/2025

Summary:
    - 
To-Do:
    - 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, Optional, Callable
import logging
import configparser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
#config = configparser.ConfigParser()
#config.read('config.ini')

class BlackScholesCalculator:
    """
    Black Scholes Calculator....
    """

    def __init__(self, S: float, K: float, T: float,
                 r: float, sigma: float, q: float = 0.0):
        self.S = S # Spot Price
        self.K = K # Strike Price
        self.T = T # Time to maturity (years)
        self.r = r # Risk-free interest rate
        self.sigma = sigma # Volatility
        self.q = q # Dividend yield

        self.validate_inputs()

    def validate_inputs(self) -> None:
        """Validates the inputs provided"""
        if any(val < 0 for val in [self.S, self.K, self.T, self.r, self.sigma, self.q]):
            raise ValueError("Negative values are not allowed for Black Scholes parameters")
        if self.sigma == 0:
            logger.warning("Zero Volatility entered - Note this may result inaccurate pricing!")

    @property
    def d1(self) -> float:
        """Calculate d1 parameter"""
        return (np.log(self.S / self.K) + 
                (self.r - self.q + 0.5 *self.sigma**2) * self.T) / \
                (self.sigma * np.sqrt(self.T))
    
    @property
    def d2(self) -> float:
        "calculates d2 parameter"
        return self.d1 - self.sigma * np.sqrt(self.T)

    def price(self) -> Tuple[float, float]:
        "Calculates call and out prices"
        call = (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) -
                self.K * np.exp(-self.r *self.T) * norm.cdf(self.d2))
        put = (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
               self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

        return call, put
    
    #### Calculates Greeks ###
    def delta(self) -> Tuple[float, float]:
        """Calculates call and put deltas"""
        call_delta = np.exp(-self.q * self.T) * norm.cdf(self.d1)
        put_delta = np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)
        return call_delta, put_delta
    
    def gamma(self) -> float:
        """Calculate gamma (same for calls and puts)"""
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / \
               (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """Calculate vega (same for calls and puts)"""
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T)
    
class MonteCarloPricer:
    """
    Monte Carlo option pricing with geometric Brownian motion
    """


class OptionVisualizer:
    """
    Visualization tools for option pricing results
    """
    
    @staticmethod
    def plot_price_sensitivity(calculator: Callable, param_range: np.ndarray, 
                               param_name: str, option_type: str = 'call') -> None:
        """
        Plot option price sensitivity to a parameter
        
        Args:
            calculator: Function that returns option price
            param_range: Range of parameter values
            param_name: Name of parameter for labeling
            option_type: 'call' or 'put'
        """
        prices = np.zeros_like(param_range)
        for i, val in enumerate(param_range):
            prices[i] = calculator(val)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, prices)
        plt.title(f"Option Price Sensitivity to {param_name}")
        plt.xlabel(param_name)
        plt.ylabel(f"{option_type.capitalize()} Price")
        plt.grid(True)
        plt.show()
    
    def plto_monte_carlo_convergence(pricer: MonteCarloPricer, option_type: str,
                                    max_paths: int = 10_000, step_size: int = 100) -> None:
        """
        Plot Monte Carlo price convergence
        
        Args:
            pricer: MonteCarloPricer instance
            option_type: 'call' or 'put'
            max_paths: Maximum number of paths to simulate
            step_size: Step size for path increments
        """
        path_counts = np.arange(step_size, max_paths + 1, step_size)
        prices = np.zeros_like(path_counts, dtype=float)
        
        for i, n in enumerate(path_counts):
            prices[i] = pricer.price_option(option_type, n_paths=n)
        
        plt.figure(figsize=(10, 6))
        plt.plot(path_counts, prices)
        #plt.axhline(y=bs_price, color='r', linestyle='--', 
        #           label='Black-Scholes Price')
        plt.title("Monte Carlo Price Convergence")
        plt.xlabel("Number of Paths")
        plt.ylabel("Option Price")
        plt.legend()
        plt.grid(True)
        plt.show()

# For Testing
if __name__ == "__main__":
    # Configuration
    params = {
        'S': 100,    # Spot price
        'K': 100,    # Strike price
        'T': 1.0,    # Time to expiration (years)
        'r': 0.05,   # Risk-free rate
        'sigma': 0.2 # Volatility
    }

    # Black-Scholes calculation
    bs = BlackScholesCalculator(**params)
    call_price, put_price = bs.price()
    print(f"Black-Scholes Call Price: {call_price:.2f}")
    print(f"Black-Scholes Put Price: {put_price:.2f}")

    # Visualizations
    viz = OptionVisualizer()

    # Plot price vs. volatility
    def bs_vol_calculator(sigma):
        return BlackScholesCalculator(S=100, K=105, T=1.0, r=0.05, sigma=sigma).price()[0]

    volatilities = np.linspace(0.1, 0.5, 50)
    viz.plot_price_sensitivity(bs_vol_calculator, volatilities, 'Volatility')
    print("Here")
        
