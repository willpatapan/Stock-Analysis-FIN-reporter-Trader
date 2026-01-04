"""
Risk Metrics for Portfolio Analysis
Advanced risk metrics used in institutional trading
Extracted from quantitative_trading_system.py
"""

import numpy as np
from scipy.stats import norm
from typing import Literal


class RiskMetrics:
    """
    Comprehensive risk metrics for portfolio and strategy evaluation

    Includes VaR, CVaR, Sharpe ratio, Sortino ratio, Maximum Drawdown,
    and other institutional-grade risk measures.
    """

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95,
                      method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)

        VaR answers: "What is the maximum loss over a given time period at a given
        confidence level?" For example, 95% VaR of -2% means there's a 5% chance
        of losing more than 2%.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (default 0.95 for 95%)
            method: Calculation method
                - 'historical': Historical simulation (non-parametric)
                - 'parametric': Variance-Covariance method (assumes normality)
                - 'monte_carlo': Monte Carlo simulation

        Returns:
            VaR value (negative number representing potential loss)
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)

        elif method == 'parametric':
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = norm.ppf(1 - confidence_level)
            return mu + sigma * z_score

        elif method == 'monte_carlo':
            mu = np.mean(returns)
            sigma = np.std(returns)
            simulations = np.random.normal(mu, sigma, 100000)
            return np.percentile(simulations, (1 - confidence_level) * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}. Use 'historical', 'parametric', or 'monte_carlo'")

    @staticmethod
    def conditional_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR), also known as Expected Shortfall

        CVaR answers: "Given that we've breached the VaR threshold, what is the
        expected loss?" More conservative than VaR because it considers the tail
        distribution.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (default 0.95)

        Returns:
            CVaR value (expected loss in the worst scenarios)
        """
        var = RiskMetrics.value_at_risk(returns, confidence_level, method='historical')
        return returns[returns <= var].mean()

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe Ratio: Risk-adjusted return metric

        Measures excess return per unit of risk. Higher is better.
        Sharpe > 1 is good, > 2 is very good, > 3 is excellent.

        Formula: (Return - RiskFreeRate) / Volatility

        Args:
            returns: Array of returns (daily)
            risk_free_rate: Annual risk-free rate (default 0.0)

        Returns:
            Annualized Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                      target_return: float = 0.0) -> float:
        """
        Sortino Ratio: Modified Sharpe ratio using downside deviation

        Like Sharpe ratio, but only penalizes downside volatility. This is more
        appropriate since investors typically don't mind upside volatility.

        Args:
            returns: Array of returns (daily)
            risk_free_rate: Annual risk-free rate (default 0.0)
            target_return: Target or required return (default 0.0)

        Returns:
            Annualized Sortino ratio
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < target_return]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)

        if downside_std == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / downside_std

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        Maximum Drawdown: Largest peak-to-trough decline

        Measures the worst loss from a peak. Important for understanding
        worst-case scenarios and required capital buffers.

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (negative value, e.g., -0.20 for 20% drawdown)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """
        Calmar Ratio: Annual return divided by Maximum Drawdown

        Measures return per unit of drawdown risk. Higher is better.
        Popular in hedge fund analysis.

        Args:
            returns: Array of returns

        Returns:
            Calmar ratio
        """
        annual_return = np.mean(returns) * 252
        max_dd = abs(RiskMetrics.max_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0.0

    @staticmethod
    def beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Beta: Systematic risk relative to market

        Measures how much an asset moves with the market.
        Beta = 1: Moves with market
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        Beta < 0: Moves opposite to market

        Args:
            asset_returns: Array of asset returns
            market_returns: Array of market/benchmark returns (same length)

        Returns:
            Beta coefficient
        """
        if len(asset_returns) != len(market_returns):
            raise ValueError("Asset and market returns must have same length")

        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0.0

    @staticmethod
    def alpha(asset_returns: np.ndarray, market_returns: np.ndarray,
              risk_free_rate: float = 0.0) -> float:
        """
        Alpha: Excess return over what CAPM predicts

        Positive alpha means the strategy beat its expected return given its risk.
        This is the "skill" component of returns.

        Formula: Asset_Return - [RiskFree + Beta * (Market_Return - RiskFree)]

        Args:
            asset_returns: Array of asset returns (daily)
            market_returns: Array of market returns (daily)
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized alpha
        """
        beta_val = RiskMetrics.beta(asset_returns, market_returns)
        asset_annual = np.mean(asset_returns) * 252
        market_annual = np.mean(market_returns) * 252

        return asset_annual - (risk_free_rate + beta_val * (market_annual - risk_free_rate))

    @staticmethod
    def information_ratio(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Information Ratio: Excess return per unit of tracking error

        Measures how consistently a strategy beats its benchmark.
        IR > 0.5 is good, > 1.0 is excellent.

        Args:
            asset_returns: Array of strategy returns
            benchmark_returns: Array of benchmark returns

        Returns:
            Annualized information ratio
        """
        excess_returns = asset_returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / tracking_error

    @staticmethod
    def downside_deviation(returns: np.ndarray, target_return: float = 0.0) -> float:
        """
        Downside Deviation: Volatility of negative returns

        Only considers returns below the target (usually 0). Used in Sortino ratio.

        Args:
            returns: Array of returns
            target_return: Target return threshold

        Returns:
            Annualized downside deviation
        """
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.std(downside_returns)

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Win Rate: Percentage of positive return periods

        Args:
            returns: Array of returns

        Returns:
            Win rate (0.0 to 1.0)
        """
        return np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Profit Factor: Gross profit divided by gross loss

        PF > 1 means profitable overall
        PF > 1.5 is good
        PF > 2 is excellent

        Args:
            returns: Array of returns

        Returns:
            Profit factor
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        return gains / losses if losses != 0 else np.inf if gains > 0 else 0.0

    @classmethod
    def calculate_all(cls, returns: np.ndarray, risk_free_rate: float = 0.02,
                      benchmark_returns: np.ndarray = None) -> dict:
        """
        Calculate all risk metrics at once

        Args:
            returns: Array of strategy returns
            risk_free_rate: Annual risk-free rate (default 2%)
            benchmark_returns: Optional benchmark returns for beta/alpha

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'total_return': (np.prod(1 + returns) - 1) * 100,
            'annual_return': np.mean(returns) * 252 * 100,
            'annual_volatility': np.std(returns) * np.sqrt(252) * 100,
            'sharpe_ratio': cls.sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': cls.sortino_ratio(returns, risk_free_rate),
            'max_drawdown': cls.max_drawdown(returns) * 100,
            'calmar_ratio': cls.calmar_ratio(returns),
            'var_95': cls.value_at_risk(returns, 0.95) * 100,
            'cvar_95': cls.conditional_var(returns, 0.95) * 100,
            'win_rate': cls.win_rate(returns) * 100,
            'profit_factor': cls.profit_factor(returns)
        }

        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            metrics['beta'] = cls.beta(returns, benchmark_returns)
            metrics['alpha'] = cls.alpha(returns, benchmark_returns, risk_free_rate) * 100
            metrics['information_ratio'] = cls.information_ratio(returns, benchmark_returns)

        return metrics
