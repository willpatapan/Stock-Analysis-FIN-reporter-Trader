"""
Trading Signal Generator
Multi-indicator consensus-based signal generation
Extracted from trading_bot_simulation.py
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class TradingSignal:
    """Container for trading signal data"""
    action: str  # 'BUY', 'SELL', or 'HOLD'
    strength: float  # 0.0 to 1.0
    price: float
    timestamp: pd.Timestamp
    indicators: Dict[str, str]  # Indicator name -> signal direction
    buy_count: int
    sell_count: int
    total_count: int


class SignalGenerator:
    """
    Generate trading signals based on multiple technical indicators

    Uses consensus approach: combines signals from RSI, MACD, Moving Averages,
    Bollinger Bands, Stochastic, and Volume to produce final BUY/SELL/HOLD signal.
    """

    def __init__(self, consensus_threshold: float = 0.6, min_signal_strength: float = 0.3):
        """
        Initialize signal generator

        Args:
            consensus_threshold: Minimum fraction of indicators agreeing (default 0.6 = 60%)
            min_signal_strength: Minimum signal strength to act (default 0.3)
        """
        self.consensus_threshold = consensus_threshold
        self.min_signal_strength = min_signal_strength

    def generate_signal(self, row: pd.Series) -> TradingSignal:
        """
        Generate trading signal from indicator data

        Args:
            row: Series with technical indicator values
                 Must include: RSI, MACD, MACD_Signal, MACD_Diff, SMA_10, SMA_20,
                 SMA_50, Close, BB_High, BB_Low, Stoch_K, Stoch_D, Volume_Ratio

        Returns:
            TradingSignal object with action, strength, and details
        """
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        indicator_votes = {}

        # 1. RSI Signal (Relative Strength Index)
        if row['RSI'] < 30:  # Oversold
            buy_signals += 1
            indicator_votes['RSI'] = 'BUY'
        elif row['RSI'] > 70:  # Overbought
            sell_signals += 1
            indicator_votes['RSI'] = 'SELL'
        else:
            indicator_votes['RSI'] = 'NEUTRAL'
        total_signals += 1

        # 2. MACD Signal (Moving Average Convergence Divergence)
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Diff'] > 0:
            buy_signals += 1
            indicator_votes['MACD'] = 'BUY'
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Diff'] < 0:
            sell_signals += 1
            indicator_votes['MACD'] = 'SELL'
        else:
            indicator_votes['MACD'] = 'NEUTRAL'
        total_signals += 1

        # 3. Moving Average Crossover
        if row['SMA_10'] > row['SMA_20'] and row['Close'] > row['SMA_50']:
            buy_signals += 1
            indicator_votes['MA_Crossover'] = 'BUY'
        elif row['SMA_10'] < row['SMA_20'] and row['Close'] < row['SMA_50']:
            sell_signals += 1
            indicator_votes['MA_Crossover'] = 'SELL'
        else:
            indicator_votes['MA_Crossover'] = 'NEUTRAL'
        total_signals += 1

        # 4. Bollinger Bands
        if row['Close'] < row['BB_Low']:  # Price below lower band
            buy_signals += 1
            indicator_votes['Bollinger'] = 'BUY'
        elif row['Close'] > row['BB_High']:  # Price above upper band
            sell_signals += 1
            indicator_votes['Bollinger'] = 'SELL'
        else:
            indicator_votes['Bollinger'] = 'NEUTRAL'
        total_signals += 1

        # 5. Stochastic Oscillator
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']:  # Oversold with bullish crossover
            buy_signals += 1
            indicator_votes['Stochastic'] = 'BUY'
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']:  # Overbought with bearish crossover
            sell_signals += 1
            indicator_votes['Stochastic'] = 'SELL'
        else:
            indicator_votes['Stochastic'] = 'NEUTRAL'
        total_signals += 1

        # 6. Volume Confirmation
        if row['Volume_Ratio'] > 1.5:  # High volume confirms signal
            if buy_signals > sell_signals:
                buy_signals += 0.5  # Boost buy signal
                indicator_votes['Volume'] = 'BUY_CONFIRM'
            elif sell_signals > buy_signals:
                sell_signals += 0.5  # Boost sell signal
                indicator_votes['Volume'] = 'SELL_CONFIRM'
            else:
                indicator_votes['Volume'] = 'NEUTRAL'
        else:
            indicator_votes['Volume'] = 'LOW'
        total_signals += 1

        # Calculate signal strength
        buy_strength = buy_signals / total_signals
        sell_strength = sell_signals / total_signals

        # Determine action based on consensus
        if buy_strength >= self.consensus_threshold and buy_strength >= self.min_signal_strength:
            action = 'BUY'
            strength = buy_strength
        elif sell_strength >= self.consensus_threshold and sell_strength >= self.min_signal_strength:
            action = 'SELL'
            strength = sell_strength
        else:
            action = 'HOLD'
            strength = max(buy_strength, sell_strength)

        return TradingSignal(
            action=action,
            strength=strength,
            price=row['Close'],
            timestamp=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
            indicators=indicator_votes,
            buy_count=int(buy_signals),
            sell_count=int(sell_signals),
            total_count=total_signals
        )

    def generate_signals_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for entire DataFrame

        Args:
            df: DataFrame with technical indicators

        Returns:
            DataFrame with additional columns: Signal, Signal_Strength
        """
        signals = []
        strengths = []

        for idx, row in df.iterrows():
            signal = self.generate_signal(row)
            signals.append(signal.action)
            strengths.append(signal.strength)

        result = df.copy()
        result['Signal'] = signals
        result['Signal_Strength'] = strengths

        return result

    def calculate_position_size(self, signal_strength: float, max_position_pct: float = 0.2,
                                min_position_pct: float = 0.05) -> float:
        """
        Calculate position size based on signal strength

        Stronger signals get larger position sizes.

        Args:
            signal_strength: Signal strength (0.0 to 1.0)
            max_position_pct: Maximum position as % of portfolio (default 20%)
            min_position_pct: Minimum position as % of portfolio (default 5%)

        Returns:
            Position size as percentage of portfolio
        """
        position_range = max_position_pct - min_position_pct
        position_size = min_position_pct + (signal_strength * position_range)
        return np.clip(position_size, min_position_pct, max_position_pct)


class StrategySignals:
    """
    Specific trading strategy signal generators

    Provides signals for various well-known trading strategies.
    """

    @staticmethod
    def rsi_mean_reversion(rsi: float, oversold: float = 30, overbought: float = 70) -> str:
        """
        RSI mean reversion signal

        Args:
            rsi: Current RSI value
            oversold: Oversold threshold (default 30)
            overbought: Overbought threshold (default 70)

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if rsi < oversold:
            return 'BUY'
        elif rsi > overbought:
            return 'SELL'
        return 'HOLD'

    @staticmethod
    def macd_crossover(macd: float, signal: float, prev_macd: float, prev_signal: float) -> str:
        """
        MACD crossover signal

        Args:
            macd: Current MACD value
            signal: Current signal line value
            prev_macd: Previous MACD value
            prev_signal: Previous signal line value

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_signal and macd > signal:
            return 'BUY'
        # Bearish crossover: MACD crosses below signal
        elif prev_macd >= prev_signal and macd < signal:
            return 'SELL'
        return 'HOLD'

    @staticmethod
    def moving_average_crossover(fast_ma: float, slow_ma: float,
                                 prev_fast_ma: float, prev_slow_ma: float) -> str:
        """
        Moving average crossover signal (Golden Cross / Death Cross)

        Args:
            fast_ma: Current fast MA value
            slow_ma: Current slow MA value
            prev_fast_ma: Previous fast MA value
            prev_slow_ma: Previous slow MA value

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Golden Cross: Fast MA crosses above slow MA
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            return 'BUY'
        # Death Cross: Fast MA crosses below slow MA
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            return 'SELL'
        return 'HOLD'

    @staticmethod
    def bollinger_breakout(close: float, bb_upper: float, bb_lower: float, bb_mid: float) -> str:
        """
        Bollinger Bands breakout signal

        Args:
            close: Current close price
            bb_upper: Upper Bollinger Band
            bb_lower: Lower Bollinger Band
            bb_mid: Middle Bollinger Band

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Mean reversion strategy
        if close < bb_lower:
            return 'BUY'  # Expecting bounce back
        elif close > bb_upper:
            return 'SELL'  # Expecting pullback
        return 'HOLD'

    @staticmethod
    def momentum_strategy(roc: float, threshold: float = 2.0) -> str:
        """
        Momentum strategy signal

        Args:
            roc: Rate of Change (%)
            threshold: Threshold for strong momentum (default 2%)

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if roc > threshold:
            return 'BUY'  # Strong positive momentum
        elif roc < -threshold:
            return 'SELL'  # Strong negative momentum
        return 'HOLD'

    @staticmethod
    def trend_following(close: float, sma_50: float, sma_200: float) -> str:
        """
        Trend following signal

        Args:
            close: Current close price
            sma_50: 50-day SMA
            sma_200: 200-day SMA

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Strong uptrend: Price above both MAs and 50 > 200
        if close > sma_50 > sma_200:
            return 'BUY'
        # Strong downtrend: Price below both MAs and 50 < 200
        elif close < sma_50 < sma_200:
            return 'SELL'
        return 'HOLD'
