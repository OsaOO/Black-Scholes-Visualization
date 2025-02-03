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
config = configparser.ConfigParser()
config.read('config.ini')

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

        
        
