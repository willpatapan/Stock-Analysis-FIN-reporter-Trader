"""
Black-Scholes-Merton Option Pricing Model
Industry standard for European options pricing
Extracted from quantitative_trading_system.py
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing model with Greeks.

    This model prices European call and put options and calculates
    the sensitivities (Greeks) used for risk management.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate d1 parameter of Black-Scholes formula

        Args:
            S: Spot price of the underlying asset
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free interest rate (annual)
            sigma: Volatility (annual)
            q: Dividend yield (annual, default 0)

        Returns:
            d1 parameter
        """
        return (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate d2 parameter of Black-Scholes formula

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            d2 parameter
        """
        return BlackScholesModel.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        European call option price using Black-Scholes formula

        Args:
            S: Spot price of underlying
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free interest rate (annual)
            sigma: Volatility (annual)
            q: Dividend yield (annual, default 0)

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        European put option price using Black-Scholes formula

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0,
              option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Delta: Rate of change of option price with respect to underlying price

        Measures how much the option price changes for a $1 change in the underlying.
        Call delta: 0 to 1
        Put delta: -1 to 0

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'

        Returns:
            Delta value
        """
        if T <= 0:
            return 1.0 if option_type == 'call' and S > K else 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)

        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Gamma: Rate of change of delta with respect to underlying price

        Measures the curvature of the option value. High gamma means delta
        changes rapidly, requiring frequent hedging.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Gamma value (same for calls and puts)
        """
        if T <= 0:
            return 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Vega: Sensitivity to volatility (per 1% change)

        Measures how much the option price changes for a 1% change in volatility.
        Vega is highest for at-the-money options.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Vega value (same for calls and puts)
        """
        if T <= 0:
            return 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0,
              option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Theta: Time decay (per day)

        Measures how much the option value decreases as time passes (holding
        everything else constant). Usually negative for long options.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'

        Returns:
            Theta value (per day)
        """
        if T <= 0:
            return 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        term1 = -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))

        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
            return (term1 + term2 + term3) / 365
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            return (term1 + term2 + term3) / 365

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, q: float = 0,
            option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Rho: Sensitivity to interest rate (per 1% change)

        Measures how much the option price changes for a 1% change in the
        risk-free interest rate. Less important for short-dated options.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'

        Returns:
            Rho value
        """
        if T <= 0:
            return 0.0

        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        if option_type == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    @classmethod
    def all_greeks(cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0,
                   option_type: Literal['call', 'put'] = 'call') -> dict:
        """
        Calculate all Greeks at once for efficiency

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: 'call' or 'put'

        Returns:
            Dictionary with all Greeks and option price
        """
        if option_type == 'call':
            price = cls.call_price(S, K, T, r, sigma, q)
        else:
            price = cls.put_price(S, K, T, r, sigma, q)

        return {
            'price': price,
            'delta': cls.delta(S, K, T, r, sigma, q, option_type),
            'gamma': cls.gamma(S, K, T, r, sigma, q),
            'vega': cls.vega(S, K, T, r, sigma, q),
            'theta': cls.theta(S, K, T, r, sigma, q, option_type),
            'rho': cls.rho(S, K, T, r, sigma, q, option_type)
        }
