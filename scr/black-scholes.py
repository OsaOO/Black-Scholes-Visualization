"""
Author: OsaO
Last Updated By: OsaO
Last Updated Date: 03/02/2025

Summary:
    - 
To-Do:
    - 
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
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
            raise ValueError("Negative vualues are not allowed for Black Scholes parameters")
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
    def plot_price_sensitivity(calculator: BlackScholesCalculator, 
                              S_range: np.ndarray, 
                              sigma_range: np.ndarray,
                              purchase_price: Optional[float] = None,
                              precision: int = 2) -> None:
        """
        Plot option price sensitivity to a parameter
        
        Args:
            calculator: Function that returns option price
            S_range: Range of stock price
            sigma_range: Range of Volatility values
            precision: Number of decimal places for annotations
            purchase_price: Optional purchase price for P&L calculation
            param_range: Range of parameter values
            param_name: Name of parameter for labeling
            option_type: 'call' or 'put'
        """

        # Initialize price matrices
        call_prices = np.zeros((len(sigma_range), len(S_range)))
        put_prices = np.zeros((len(sigma_range), len(S_range)))

        # Calculate prices for each combination
        for i, sigma in enumerate(sigma_range):
            for j, S in enumerate(S_range):
                bs = BlackScholesCalculator(
                    S=S,
                    K=calculator.K,
                    T=calculator.T,
                    r=calculator.r,
                    sigma=sigma,
                    q=calculator.q
                )
                call, put = bs.price()
                
                # Makes adjustment depending on if a purchase price is provided
                if purchase_price is not None:
                    call_prices[i, j] = call - purchase_price
                    put_prices[i, j] = put - purchase_price
                else:
                    call_prices[i, j] = call
                    put_prices[i, j] = put

        # Create DataFrames for heatmaps
        call_df = pd.DataFrame(call_prices, 
                             index=np.round(sigma_range, 2),  # y-axis labels
                             columns=np.round(S_range, 2))    # x-axis labels
        put_df = pd.DataFrame(put_prices,
                            index=np.round(sigma_range, 2),
                            columns=np.round(S_range, 2))
        
        # Configure visualization parameters
        is_pl = purchase_price is not None
        title_suffix = "P&L" if is_pl else "Prices"
        #fmt = f"+.{precision}f" if is_pl else f".{precision}f" # If want to change percisiioin for price & PnL
        #cmap_center = 0 if is_pl else None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Black-Scholes Option {title_suffix} Heatmaps', fontsize=4, y=1.02)

        # Common heatmap parameters
        heatmap_kwargs = {
            'annot': True,
            'fmt': f'.{precision}f',
            'cmap': 'RdYlGn',
            'linewidths': 0.5,
            'linecolor': 'white',
            'cbar': True,
            'annot_kws': {'fontsize': 8}
        }

        # Plot call heatmap
        sns.heatmap(call_df, ax=ax1, **heatmap_kwargs)
        ax1.set_title(f'Call Option {title_suffix}', pad=20)
        ax1.set_xlabel('Stock Price', labelpad=10)
        ax1.set_ylabel('Volatility', labelpad=10)
        ax1.invert_yaxis()  # Ensure volatility increases from bottom to top

        # Plot put heatmap
        sns_put = sns.heatmap(put_df, ax=ax2, **heatmap_kwargs)
        ax2.set_title(f'Put Option {title_suffix}', pad=20)
        ax2.set_xlabel('Stock Price', labelpad=10)
        ax2.set_ylabel('Volatility', labelpad=10)
        ax2.invert_yaxis()

        # Adjust colorbar labels
        ax1.collections[0].colorbar.set_label(f'{title_suffix}', rotation=270, labelpad=20)
        ax2.collections[0].colorbar.set_label(f'{title_suffix}', rotation=270, labelpad=20)

        plt.tight_layout()
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
    entryPrice = 3.25

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
    
    # Create base calculator with default parameters
    base_calculator = BlackScholesCalculator(
        S=100,  # This will be overridden in ranges
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
        q=0
    )

    # Generate ranges
    S_range = np.linspace(80, 120, 10)  # Stock prices e.g. from 80 to 120
    sigma_range = np.linspace(0.1, 0.5, 10)  # Volatility e.g. from 10% to 50%

    # Plot heatmaps
    OptionVisualizer.plot_price_sensitivity(
        calculator=base_calculator,
        S_range=S_range,
        sigma_range=sigma_range,
        purchase_price=5.0
    )
    #volatilities = np.linspace(0.1, 0.5, 50)
    #viz.plot_price_sensitivity(bs_vol_calculator, volatilities, 'Volatility')
    print("Here")
        
