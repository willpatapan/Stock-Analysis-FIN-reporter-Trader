"""
Portfolio Optimization using Modern Portfolio Theory
Implements various optimization methods for asset allocation
Extracted from quantitative_trading_system.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple
from config.settings import Config


class PortfolioOptimizer:
    """
    Modern Portfolio Theory (MPT) implementation

    Provides various portfolio optimization methods:
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Risk Parity
    - Efficient Frontier generation
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize optimizer with historical returns

        Args:
            returns: DataFrame with assets as columns, dates as index
                    Returns should be daily percentage returns (e.g., 0.01 for 1%)
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()

    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio return and volatility for given weights

        Args:
            weights: Array of portfolio weights (must sum to 1)

        Returns:
            Tuple of (annualized_return, annualized_volatility)
        """
        portfolio_return = np.sum(self.mean_returns * weights) * 252  # Annualize
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std

    def negative_sharpe(self, weights: np.ndarray, risk_free_rate: float = None) -> float:
        """
        Negative Sharpe ratio for minimization

        Args:
            weights: Portfolio weights
            risk_free_rate: Annual risk-free rate (defaults to Config value)

        Returns:
            Negative Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = Config.DEFAULT_RISK_FREE_RATE

        p_return, p_std = self.portfolio_stats(weights)

        if p_std == 0:
            return np.inf

        return -(p_return - risk_free_rate) / p_std

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Portfolio variance for minimum variance optimization

        Args:
            weights: Portfolio weights

        Returns:
            Annualized portfolio variance
        """
        return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))

    def max_sharpe_portfolio(self, risk_free_rate: float = None) -> Dict:
        """
        Find portfolio with maximum Sharpe ratio

        This is the "tangency portfolio" in MPT - the portfolio with the
        best risk-adjusted returns.

        Args:
            risk_free_rate: Annual risk-free rate (defaults to Config value)

        Returns:
            Dictionary with:
                - weights: Optimal weights array
                - return: Expected annual return
                - volatility: Expected annual volatility
                - sharpe_ratio: Sharpe ratio
                - asset_weights: Dict mapping asset names to weights
        """
        if risk_free_rate is None:
            risk_free_rate = Config.DEFAULT_RISK_FREE_RATE

        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Bounds: long-only (0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Initial guess: equal weights
        initial_guess = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(
            self.negative_sharpe,
            initial_guess,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        ret, vol = self.portfolio_stats(weights)
        sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0

        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'asset_weights': dict(zip(self.asset_names, weights))
        }

    def min_variance_portfolio(self) -> Dict:
        """
        Find minimum variance portfolio

        This is the portfolio with the lowest risk, regardless of return.
        Useful for very conservative investors.

        Returns:
            Dictionary with:
                - weights: Optimal weights array
                - return: Expected annual return
                - volatility: Expected annual volatility (minimized)
                - asset_weights: Dict mapping asset names to weights
        """
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(
            self.portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        ret, vol = self.portfolio_stats(weights)

        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'asset_weights': dict(zip(self.asset_names, weights))
        }

    def efficient_frontier(self, n_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier

        The efficient frontier is the set of optimal portfolios that offer
        the highest expected return for a given level of risk.

        Args:
            n_portfolios: Number of portfolios to generate (default 50)

        Returns:
            DataFrame with columns: return, volatility, weights
        """
        min_ret = self.mean_returns.min() * 252
        max_ret = self.mean_returns.max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)

        results = []

        for target in target_returns:
            # Constraints: weights sum to 1, target return achieved
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_stats(x)[0] - target}
            )
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)

            result = minimize(
                self.portfolio_variance,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                weights = result.x
                ret, vol = self.portfolio_stats(weights)
                results.append({
                    'return': ret,
                    'volatility': vol,
                    'weights': weights
                })

        return pd.DataFrame(results)

    def risk_parity_portfolio(self) -> Dict:
        """
        Risk Parity portfolio: Equal risk contribution from each asset

        Unlike equal-weight or market-cap weighting, risk parity allocates
        capital so each asset contributes equally to portfolio risk.
        Popular in institutional investing.

        Returns:
            Dictionary with:
                - weights: Optimal weights array
                - return: Expected annual return
                - volatility: Expected annual volatility
                - asset_weights: Dict mapping asset names to weights
                - risk_contributions: Risk contribution from each asset
        """
        def risk_contribution(weights):
            """Calculate risk contribution from each asset"""
            portfolio_vol = np.sqrt(self.portfolio_variance(weights))
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            contrib = weights * marginal_contrib / portfolio_vol if portfolio_vol != 0 else weights * 0
            return contrib

        def risk_parity_objective(weights):
            """Minimize squared differences from equal risk contribution"""
            portfolio_vol = np.sqrt(self.portfolio_variance(weights))
            contrib = risk_contribution(weights)
            target_contrib = portfolio_vol / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.001, 1) for _ in range(self.n_assets))  # Small lower bound to avoid division issues
        initial_guess = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(
            risk_parity_objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol = self.portfolio_stats(weights)
        risk_contrib = risk_contribution(weights)

        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'asset_weights': dict(zip(self.asset_names, weights)),
            'risk_contributions': dict(zip(self.asset_names, risk_contrib))
        }

    def target_return_portfolio(self, target_return: float) -> Dict:
        """
        Find minimum variance portfolio for a target return

        Args:
            target_return: Desired annual return (e.g., 0.10 for 10%)

        Returns:
            Dictionary with portfolio details, or None if infeasible
        """
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_stats(x)[0] - target_return}
        )
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(
            self.portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return None

        weights = result.x
        ret, vol = self.portfolio_stats(weights)

        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'asset_weights': dict(zip(self.asset_names, weights))
        }

    def target_volatility_portfolio(self, target_volatility: float,
                                    risk_free_rate: float = None) -> Dict:
        """
        Find maximum return portfolio for a target volatility

        Args:
            target_volatility: Desired annual volatility (e.g., 0.15 for 15%)
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary with portfolio details, or None if infeasible
        """
        if risk_free_rate is None:
            risk_free_rate = Config.DEFAULT_RISK_FREE_RATE

        def negative_return(weights):
            ret, _ = self.portfolio_stats(weights)
            return -ret

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_stats(x)[1] - target_volatility}
        )
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)

        result = minimize(
            negative_return,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return None

        weights = result.x
        ret, vol = self.portfolio_stats(weights)
        sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0

        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'asset_weights': dict(zip(self.asset_names, weights))
        }

    def backtest_portfolio(self, weights: np.ndarray, initial_value: float = 10000) -> pd.Series:
        """
        Backtest a portfolio with given weights

        Args:
            weights: Portfolio weights
            initial_value: Starting portfolio value (default $10,000)

        Returns:
            Series of portfolio values over time
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        portfolio_value = initial_value * (1 + portfolio_returns).cumprod()
        return portfolio_value
