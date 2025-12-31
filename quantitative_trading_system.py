"""
Advanced Quantitative Trading System
Implements Wall Street-grade derivatives pricing, risk management, and portfolio optimization.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ==================== DERIVATIVES PRICING ====================

class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing model with Greeks.
    Industry standard for European options pricing.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d1 parameter"""
        return (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d2 parameter"""
        return BlackScholesModel.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        European call option price
        S: Spot price, K: Strike, T: Time to maturity, r: Risk-free rate,
        sigma: Volatility, q: Dividend yield
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """European put option price"""
        if T <= 0:
            return max(K - S, 0)

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0,
              option_type: str = 'call') -> float:
        """Delta: Rate of change of option price with respect to underlying price"""
        if T <= 0:
            return 1.0 if option_type == 'call' and S > K else 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)

        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Gamma: Rate of change of delta with respect to underlying price"""
        if T <= 0:
            return 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Vega: Sensitivity to volatility (per 1% change)"""
        if T <= 0:
            return 0.0

        d1 = BlackScholesModel.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0,
              option_type: str = 'call') -> float:
        """Theta: Time decay (per day)"""
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
            option_type: str = 'call') -> float:
        """Rho: Sensitivity to interest rate (per 1% change)"""
        if T <= 0:
            return 0.0

        d2 = BlackScholesModel.d2(S, K, T, r, sigma, q)

        if option_type == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


class ImpliedVolatility:
    """Calculate implied volatility from market prices using Newton-Raphson method"""

    @staticmethod
    def calculate(market_price: float, S: float, K: float, T: float, r: float,
                  option_type: str = 'call', q: float = 0, max_iter: int = 100,
                  tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        if T <= 0:
            return np.nan

        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
        sigma = max(0.01, min(sigma, 5.0))  # Bound between 1% and 500%

        for i in range(max_iter):
            if option_type == 'call':
                price = BlackScholesModel.call_price(S, K, T, r, sigma, q)
            else:
                price = BlackScholesModel.put_price(S, K, T, r, sigma, q)

            vega = BlackScholesModel.vega(S, K, T, r, sigma, q) * 100  # Convert back to full vega

            diff = market_price - price

            if abs(diff) < tolerance:
                return sigma

            if vega == 0:
                return np.nan

            sigma = sigma + diff / vega
            sigma = max(0.001, min(sigma, 5.0))  # Keep sigma bounded

        return sigma


# ==================== RISK MANAGEMENT ====================

class RiskMetrics:
    """
    Advanced risk metrics used in institutional trading.
    """

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)

        methods:
        - historical: Historical simulation
        - parametric: Variance-Covariance method
        - monte_carlo: Monte Carlo simulation
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
            raise ValueError(f"Unknown VaR method: {method}")

    @staticmethod
    def conditional_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR/Expected Shortfall)
        Expected loss given that VaR threshold is exceeded
        """
        var = RiskMetrics.value_at_risk(returns, confidence_level, method='historical')
        return returns[returns <= var].mean()

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Sharpe Ratio: Risk-adjusted return"""
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                      target_return: float = 0.0) -> float:
        """Sortino Ratio: Sharpe ratio using downside deviation"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < target_return]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        return np.sqrt(252) * np.mean(excess_returns) / downside_std

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Maximum Drawdown: Largest peak-to-trough decline"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calmar Ratio: Annual return / Maximum Drawdown"""
        annual_return = np.mean(returns) * 252
        max_dd = abs(RiskMetrics.max_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0.0

    @staticmethod
    def beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Beta: Systematic risk relative to market"""
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0.0


# ==================== PORTFOLIO OPTIMIZATION ====================

class PortfolioOptimizer:
    """
    Modern Portfolio Theory implementation with various optimization methods.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with historical returns DataFrame
        returns: DataFrame with assets as columns, dates as index
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)

    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio return and volatility"""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std

    def negative_sharpe(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Negative Sharpe ratio for minimization"""
        p_return, p_std = self.portfolio_stats(weights)
        return -(p_return - risk_free_rate) / p_std

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """Portfolio variance for minimum variance optimization"""
        return np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))

    def max_sharpe_portfolio(self, risk_free_rate: float = 0.02) -> Dict:
        """Find portfolio with maximum Sharpe ratio"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
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
        sharpe = (ret - risk_free_rate) / vol

        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }

    def min_variance_portfolio(self) -> Dict:
        """Find minimum variance portfolio"""
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
            'volatility': vol
        }

    def efficient_frontier(self, n_portfolios: int = 50) -> pd.DataFrame:
        """Generate efficient frontier"""
        min_ret = self.mean_returns.min() * 252
        max_ret = self.mean_returns.max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)

        results = []

        for target in target_returns:
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
                constraints=constraints
            )

            if result.success:
                weights = result.x
                ret, vol = self.portfolio_stats(weights)
                results.append({'return': ret, 'volatility': vol, 'weights': weights})

        return pd.DataFrame(results)

    def risk_parity_portfolio(self) -> Dict:
        """
        Risk Parity portfolio: Equal risk contribution from each asset
        """
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(self.portfolio_variance(weights))
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib

        def risk_parity_objective(weights):
            contrib = risk_contribution(weights)
            target_contrib = portfolio_vol / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)

        # Calculate portfolio vol for target
        temp_vol = np.sqrt(self.portfolio_variance(initial_guess))
        portfolio_vol = temp_vol

        result = minimize(
            risk_parity_objective,
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
            'volatility': vol
        }


# ==================== VOLATILITY MODELS ====================

class VolatilityModels:
    """
    Advanced volatility forecasting models used in derivatives trading.
    """

    @staticmethod
    def ewma_volatility(returns: np.ndarray, lambda_param: float = 0.94) -> np.ndarray:
        """
        Exponentially Weighted Moving Average (EWMA) volatility
        RiskMetrics standard: lambda = 0.94 for daily data
        """
        squared_returns = returns ** 2
        ewma_var = np.zeros(len(returns))
        ewma_var[0] = squared_returns[0]

        for t in range(1, len(returns)):
            ewma_var[t] = lambda_param * ewma_var[t-1] + (1 - lambda_param) * squared_returns[t]

        return np.sqrt(ewma_var)

    @staticmethod
    def garch_volatility(returns: np.ndarray, omega: float = 0.00001,
                         alpha: float = 0.08, beta: float = 0.90) -> np.ndarray:
        """
        GARCH(1,1) volatility model
        Industry standard parameters: omega=1e-5, alpha=0.08, beta=0.90
        """
        squared_returns = returns ** 2
        variance = np.zeros(len(returns))
        variance[0] = squared_returns[0]

        for t in range(1, len(returns)):
            variance[t] = omega + alpha * squared_returns[t-1] + beta * variance[t-1]

        return np.sqrt(variance)

    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """
        Parkinson's range-based volatility estimator
        More efficient than close-to-close volatility
        """
        return np.sqrt((1 / (4 * np.log(2))) * np.log(high / low) ** 2)

    @staticmethod
    def garman_klass_volatility(open_price: np.ndarray, high: np.ndarray,
                                low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Garman-Klass volatility estimator using OHLC data
        More efficient than Parkinson estimator
        """
        term1 = 0.5 * np.log(high / low) ** 2
        term2 = (2 * np.log(2) - 1) * np.log(close / open_price) ** 2
        return np.sqrt(term1 - term2)


# ==================== TRADING STRATEGIES ====================

class QuantitativeStrategies:
    """
    Advanced quantitative trading strategies.
    """

    @staticmethod
    def mean_reversion_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Z-score based mean reversion signal
        Signal > 2: Overbought (sell)
        Signal < -2: Oversold (buy)
        """
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        zscore = (prices - rolling_mean) / rolling_std
        return zscore

    @staticmethod
    def pairs_trading_signal(price1: pd.Series, price2: pd.Series,
                            window: int = 20) -> pd.Series:
        """
        Pairs trading signal using cointegration spread
        """
        # Calculate hedge ratio using OLS
        beta = np.cov(price1, price2)[0, 1] / np.var(price2)
        spread = price1 - beta * price2

        # Generate z-score signal
        return QuantitativeStrategies.mean_reversion_zscore(spread, window)

    @staticmethod
    def momentum_signal(returns: pd.Series, lookback: int = 20) -> pd.Series:
        """
        Momentum signal based on past returns
        Positive momentum: Buy
        Negative momentum: Sell
        """
        momentum = returns.rolling(window=lookback).sum()
        return momentum

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        RSI > 70: Overbought
        RSI < 30: Oversold
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20,
                       num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        Returns: (middle_band, upper_band, lower_band)
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle, upper, lower


# ==================== EXAMPLE USAGE ====================

def main():
    """Demonstrate the quantitative trading system"""

    print("=" * 80)
    print("WALL STREET QUANTITATIVE TRADING SYSTEM")
    print("=" * 80)

    # ===== BLACK-SCHOLES OPTIONS PRICING =====
    print("\n1. BLACK-SCHOLES OPTIONS PRICING & GREEKS")
    print("-" * 80)

    S = 100    # Spot price
    K = 100    # Strike price
    T = 1.0    # 1 year to maturity
    r = 0.05   # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    q = 0.02   # 2% dividend yield

    call_price = BlackScholesModel.call_price(S, K, T, r, sigma, q)
    put_price = BlackScholesModel.put_price(S, K, T, r, sigma, q)

    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price:  ${put_price:.4f}")
    print(f"Put-Call Parity Check: {abs((call_price - put_price) - (S*np.exp(-q*T) - K*np.exp(-r*T))) < 0.01}")

    print("\nThe Greeks (Call Option):")
    print(f"  Delta:   {BlackScholesModel.delta(S, K, T, r, sigma, q, 'call'):.4f}")
    print(f"  Gamma:   {BlackScholesModel.gamma(S, K, T, r, sigma, q):.4f}")
    print(f"  Vega:    {BlackScholesModel.vega(S, K, T, r, sigma, q):.4f}")
    print(f"  Theta:   {BlackScholesModel.theta(S, K, T, r, sigma, q, 'call'):.4f}")
    print(f"  Rho:     {BlackScholesModel.rho(S, K, T, r, sigma, q, 'call'):.4f}")

    # ===== IMPLIED VOLATILITY =====
    print("\n2. IMPLIED VOLATILITY CALCULATION")
    print("-" * 80)

    market_price = 10.45
    iv = ImpliedVolatility.calculate(market_price, S, K, T, r, 'call', q)
    print(f"Market Price: ${market_price:.2f}")
    print(f"Implied Volatility: {iv*100:.2f}%")

    # ===== RISK METRICS =====
    print("\n3. RISK METRICS")
    print("-" * 80)

    # Simulate returns
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns for 1 year

    var_95 = RiskMetrics.value_at_risk(returns, 0.95)
    cvar_95 = RiskMetrics.conditional_var(returns, 0.95)
    sharpe = RiskMetrics.sharpe_ratio(returns, 0.02)
    sortino = RiskMetrics.sortino_ratio(returns, 0.02)
    max_dd = RiskMetrics.max_drawdown(returns)
    calmar = RiskMetrics.calmar_ratio(returns)

    print(f"Value at Risk (95%):        {var_95*100:.2f}%")
    print(f"Conditional VaR (95%):      {cvar_95*100:.2f}%")
    print(f"Sharpe Ratio:               {sharpe:.4f}")
    print(f"Sortino Ratio:              {sortino:.4f}")
    print(f"Maximum Drawdown:           {max_dd*100:.2f}%")
    print(f"Calmar Ratio:               {calmar:.4f}")

    # ===== PORTFOLIO OPTIMIZATION =====
    print("\n4. PORTFOLIO OPTIMIZATION")
    print("-" * 80)

    # Create sample returns for 4 assets
    n_days = 252
    n_assets = 4
    asset_names = ['Tech Stock', 'Financial Stock', 'Energy Stock', 'Utility Stock']

    returns_data = pd.DataFrame(
        np.random.multivariate_normal(
            [0.0005, 0.0003, 0.0004, 0.0002],  # Expected returns
            [[0.0004, 0.0001, 0.0002, 0.00005],  # Covariance matrix
             [0.0001, 0.0002, 0.00008, 0.00003],
             [0.0002, 0.00008, 0.0003, 0.00006],
             [0.00005, 0.00003, 0.00006, 0.0001]],
            n_days
        ),
        columns=asset_names
    )

    optimizer = PortfolioOptimizer(returns_data)

    # Maximum Sharpe Ratio Portfolio
    max_sharpe = optimizer.max_sharpe_portfolio()
    print("Maximum Sharpe Ratio Portfolio:")
    print(f"  Expected Return:  {max_sharpe['return']*100:.2f}%")
    print(f"  Volatility:       {max_sharpe['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio:     {max_sharpe['sharpe_ratio']:.4f}")
    print("  Weights:")
    for name, weight in zip(asset_names, max_sharpe['weights']):
        print(f"    {name}: {weight*100:.2f}%")

    # Minimum Variance Portfolio
    print("\nMinimum Variance Portfolio:")
    min_var = optimizer.min_variance_portfolio()
    print(f"  Expected Return:  {min_var['return']*100:.2f}%")
    print(f"  Volatility:       {min_var['volatility']*100:.2f}%")
    print("  Weights:")
    for name, weight in zip(asset_names, min_var['weights']):
        print(f"    {name}: {weight*100:.2f}%")

    # Risk Parity Portfolio
    print("\nRisk Parity Portfolio:")
    risk_parity = optimizer.risk_parity_portfolio()
    print(f"  Expected Return:  {risk_parity['return']*100:.2f}%")
    print(f"  Volatility:       {risk_parity['volatility']*100:.2f}%")
    print("  Weights:")
    for name, weight in zip(asset_names, risk_parity['weights']):
        print(f"    {name}: {weight*100:.2f}%")

    # ===== VOLATILITY MODELS =====
    print("\n5. VOLATILITY FORECASTING")
    print("-" * 80)

    sample_returns = returns[-30:]  # Last 30 days

    ewma_vol = VolatilityModels.ewma_volatility(sample_returns)
    garch_vol = VolatilityModels.garch_volatility(sample_returns)

    print(f"Current EWMA Volatility:  {ewma_vol[-1]*np.sqrt(252)*100:.2f}% (annualized)")
    print(f"Current GARCH Volatility: {garch_vol[-1]*np.sqrt(252)*100:.2f}% (annualized)")

    # ===== TRADING SIGNALS =====
    print("\n6. TRADING SIGNALS")
    print("-" * 80)

    # Generate sample price data
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))

    zscore = QuantitativeStrategies.mean_reversion_zscore(prices, window=20)
    momentum = QuantitativeStrategies.momentum_signal(pd.Series(returns), lookback=20)
    rsi = QuantitativeStrategies.rsi(prices, period=14)

    print(f"Current Z-Score:          {zscore.iloc[-1]:.2f}")
    print(f"Current Momentum:         {momentum.iloc[-1]*100:.4f}%")
    print(f"Current RSI:              {rsi.iloc[-1]:.2f}")

    if zscore.iloc[-1] > 2:
        print("Mean Reversion Signal:    SELL (Overbought)")
    elif zscore.iloc[-1] < -2:
        print("Mean Reversion Signal:    BUY (Oversold)")
    else:
        print("Mean Reversion Signal:    NEUTRAL")

    if rsi.iloc[-1] > 70:
        print("RSI Signal:               SELL (Overbought)")
    elif rsi.iloc[-1] < 30:
        print("RSI Signal:               BUY (Oversold)")
    else:
        print("RSI Signal:               NEUTRAL")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
