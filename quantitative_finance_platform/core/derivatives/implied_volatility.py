"""
Implied Volatility Calculator
Uses Newton-Raphson method to back out volatility from market prices
Extracted from quantitative_trading_system.py
"""

import numpy as np
from typing import Literal
from core.derivatives.black_scholes import BlackScholesModel


class ImpliedVolatility:
    """
    Calculate implied volatility from market option prices

    Uses Newton-Raphson iterative method to find the volatility that
    makes the Black-Scholes theoretical price equal the market price.
    """

    @staticmethod
    def calculate(market_price: float, S: float, K: float, T: float, r: float,
                  option_type: Literal['call', 'put'] = 'call', q: float = 0,
                  max_iter: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method

        Args:
            market_price: Observed market price of the option
            S: Spot price of underlying
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free interest rate
            option_type: 'call' or 'put'
            q: Dividend yield (default 0)
            max_iter: Maximum iterations (default 100)
            tolerance: Convergence tolerance (default 1e-6)

        Returns:
            Implied volatility (annual), or np.nan if failed to converge
        """
        if T <= 0:
            return np.nan

        if market_price <= 0:
            return np.nan

        # Initial guess using Brenner-Subrahmanyam approximation
        # sigma ≈ sqrt(2π/T) * (market_price/S)
        sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
        sigma = max(0.01, min(sigma, 5.0))  # Bound between 1% and 500%

        for i in range(max_iter):
            # Calculate theoretical price with current sigma guess
            if option_type == 'call':
                price = BlackScholesModel.call_price(S, K, T, r, sigma, q)
            else:
                price = BlackScholesModel.put_price(S, K, T, r, sigma, q)

            # Calculate vega (derivative of price w.r.t. volatility)
            vega = BlackScholesModel.vega(S, K, T, r, sigma, q) * 100  # Convert to full vega

            # Difference between market and theoretical price
            diff = market_price - price

            # Check for convergence
            if abs(diff) < tolerance:
                return sigma

            # Avoid division by zero
            if vega == 0:
                return np.nan

            # Newton-Raphson update: x_new = x_old + f(x)/f'(x)
            sigma = sigma + diff / vega

            # Keep sigma in reasonable bounds
            sigma = max(0.001, min(sigma, 5.0))

        # If we reach here, convergence failed
        return sigma  # Return best estimate

    @staticmethod
    def calculate_vectorized(market_prices: np.ndarray, S: float, K: np.ndarray,
                            T: np.ndarray, r: float, option_type: Literal['call', 'put'] = 'call',
                            q: float = 0) -> np.ndarray:
        """
        Calculate implied volatility for multiple options (vectorized)

        Useful for processing entire option chains efficiently.

        Args:
            market_prices: Array of market prices
            S: Spot price (scalar)
            K: Array of strike prices
            T: Array of times to maturity
            r: Risk-free rate (scalar)
            option_type: 'call' or 'put'
            q: Dividend yield (scalar)

        Returns:
            Array of implied volatilities
        """
        results = np.zeros_like(market_prices)

        for i in range(len(market_prices)):
            results[i] = ImpliedVolatility.calculate(
                market_prices[i],
                S,
                K[i] if isinstance(K, np.ndarray) else K,
                T[i] if isinstance(T, np.ndarray) else T,
                r,
                option_type,
                q
            )

        return results

    @staticmethod
    def iv_surface(market_prices_df, S: float, r: float, q: float = 0):
        """
        Build implied volatility surface from option chain data

        Args:
            market_prices_df: DataFrame with columns ['strike', 'expiry', 'call_price', 'put_price']
            S: Spot price
            r: Risk-free rate
            q: Dividend yield

        Returns:
            DataFrame with implied volatilities for calls and puts
        """
        import pandas as pd

        results = []

        for _, row in market_prices_df.iterrows():
            K = row['strike']
            T = row['expiry']  # Should be in years

            # Calculate call IV
            if 'call_price' in row and row['call_price'] > 0:
                call_iv = ImpliedVolatility.calculate(
                    row['call_price'], S, K, T, r, 'call', q
                )
            else:
                call_iv = np.nan

            # Calculate put IV
            if 'put_price' in row and row['put_price'] > 0:
                put_iv = ImpliedVolatility.calculate(
                    row['put_price'], S, K, T, r, 'put', q
                )
            else:
                put_iv = np.nan

            results.append({
                'strike': K,
                'expiry': T,
                'call_iv': call_iv,
                'put_iv': put_iv
            })

        return pd.DataFrame(results)
