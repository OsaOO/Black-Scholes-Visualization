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
        
