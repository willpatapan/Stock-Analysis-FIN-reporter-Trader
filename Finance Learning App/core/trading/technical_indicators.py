"""
Technical Indicators for Trading Analysis
Comprehensive set of technical analysis indicators
Extracted from trading_bot_simulation.py
"""

import pandas as pd
import numpy as np
import ta


class TechnicalIndicators:
    """
    Calculate technical indicators for trading strategies

    Provides moving averages, momentum indicators, volatility measures,
    and volume analysis tools used in algorithmic trading.
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators at once

        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)

        Returns:
            DataFrame with all technical indicators added as columns
        """
        result = df.copy()

        # Moving Averages
        result['SMA_10'] = ta.trend.sma_indicator(result['Close'], window=10)
        result['SMA_20'] = ta.trend.sma_indicator(result['Close'], window=20)
        result['SMA_50'] = ta.trend.sma_indicator(result['Close'], window=50)
        result['SMA_200'] = ta.trend.sma_indicator(result['Close'], window=200)
        result['EMA_12'] = ta.trend.ema_indicator(result['Close'], window=12)
        result['EMA_26'] = ta.trend.ema_indicator(result['Close'], window=26)

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(result['Close'])
        result['MACD'] = macd.macd()
        result['MACD_Signal'] = macd.macd_signal()
        result['MACD_Diff'] = macd.macd_diff()

        # RSI (Relative Strength Index)
        result['RSI'] = ta.momentum.rsi(result['Close'], window=14)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(result['Close'])
        result['BB_High'] = bollinger.bollinger_hband()
        result['BB_Low'] = bollinger.bollinger_lband()
        result['BB_Mid'] = bollinger.bollinger_mavg()
        result['BB_Width'] = (result['BB_High'] - result['BB_Low']) / result['BB_Mid']

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(result['High'], result['Low'], result['Close'])
        result['Stoch_K'] = stoch.stoch()
        result['Stoch_D'] = stoch.stoch_signal()

        # ATR (Average True Range) - Volatility
        result['ATR'] = ta.volatility.average_true_range(result['High'], result['Low'], result['Close'])

        # Volume Indicators
        result['Volume_SMA'] = result['Volume'].rolling(window=20).mean()
        result['Volume_Ratio'] = result['Volume'] / result['Volume_SMA']
        result['OBV'] = ta.volume.on_balance_volume(result['Close'], result['Volume'])

        # Momentum Indicators
        result['ROC'] = ta.momentum.roc(result['Close'], window=10)
        result['Daily_Return'] = result['Close'].pct_change()
        result['Momentum'] = result['Close'] - result['Close'].shift(10)

        # Price Channels
        result['High_20'] = result['High'].rolling(window=20).max()
        result['Low_20'] = result['Low'].rolling(window=20).min()

        return result.dropna()

    @staticmethod
    def moving_averages(close: pd.Series, windows: list = [10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages for multiple windows

        Args:
            close: Series of closing prices
            windows: List of window sizes (default: [10, 20, 50, 200])

        Returns:
            DataFrame with SMA columns
        """
        result = pd.DataFrame(index=close.index)
        for window in windows:
            result[f'SMA_{window}'] = ta.trend.sma_indicator(close, window=window)
        return result

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            close: Series of closing prices
            fast: Fast EMA window (default 12)
            slow: Slow EMA window (default 26)
            signal: Signal line EMA window (default 9)

        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        macd_obj = ta.trend.MACD(close, window_slow=slow, window_fast=fast, window_sign=signal)
        return pd.DataFrame({
            'MACD': macd_obj.macd(),
            'MACD_Signal': macd_obj.macd_signal(),
            'MACD_Histogram': macd_obj.macd_diff()
        }, index=close.index)

    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)

        RSI > 70: Overbought
        RSI < 30: Oversold

        Args:
            close: Series of closing prices
            window: Lookback period (default 14)

        Returns:
            Series with RSI values (0-100)
        """
        return ta.momentum.rsi(close, window=window)

    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Args:
            close: Series of closing prices
            window: Moving average window (default 20)
            std_dev: Number of standard deviations (default 2)

        Returns:
            DataFrame with Upper, Middle, Lower bands and Width
        """
        bb = ta.volatility.BollingerBands(close, window=window, window_dev=std_dev)
        return pd.DataFrame({
            'BB_Upper': bb.bollinger_hband(),
            'BB_Middle': bb.bollinger_mavg(),
            'BB_Lower': bb.bollinger_lband(),
            'BB_Width': (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        }, index=close.index)

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator

        %K > 80: Overbought
        %K < 20: Oversold

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_window: %K window (default 14)
            d_window: %D smoothing window (default 3)

        Returns:
            DataFrame with %K and %D values
        """
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=k_window, smooth_window=d_window)
        return pd.DataFrame({
            'Stoch_K': stoch.stoch(),
            'Stoch_D': stoch.stoch_signal()
        }, index=close.index)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate ATR (Average True Range)

        Measures volatility. Higher ATR = higher volatility.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Lookback period (default 14)

        Returns:
            Series with ATR values
        """
        return ta.volatility.average_true_range(high, low, close, window=window)

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index)

        Measures trend strength.
        ADX > 25: Strong trend
        ADX < 20: Weak trend

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Lookback period (default 14)

        Returns:
            DataFrame with ADX, +DI, -DI
        """
        adx_obj = ta.trend.ADXIndicator(high, low, close, window=window)
        return pd.DataFrame({
            'ADX': adx_obj.adx(),
            'DI_Positive': adx_obj.adx_pos(),
            'DI_Negative': adx_obj.adx_neg()
        }, index=close.index)

    @staticmethod
    def volume_weighted_average_price(high: pd.Series, low: pd.Series,
                                       close: pd.Series, volume: pd.Series,
                                       window: int = 14) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price)

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volumes
            window: Rolling window (default 14)

        Returns:
            Series with VWAP values
        """
        return ta.volume.volume_weighted_average_price(high, low, close, volume, window=window)

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate OBV (On Balance Volume)

        Cumulative volume indicator that shows buying/selling pressure.

        Args:
            close: Series of closing prices
            volume: Series of volumes

        Returns:
            Series with OBV values
        """
        return ta.volume.on_balance_volume(close, volume)

    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series,
                       conversion_window: int = 9,
                       base_window: int = 26,
                       span_b_window: int = 52,
                       displacement: int = 26) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components

        Args:
            high: Series of high prices
            low: Series of low prices
            conversion_window: Tenkan-sen window (default 9)
            base_window: Kijun-sen window (default 26)
            span_b_window: Senkou Span B window (default 52)
            displacement: Cloud displacement (default 26)

        Returns:
            DataFrame with Ichimoku components
        """
        ichimoku = ta.trend.IchimokuIndicator(
            high, low,
            window1=conversion_window,
            window2=base_window,
            window3=span_b_window
        )

        return pd.DataFrame({
            'Tenkan_sen': ichimoku.ichimoku_conversion_line(),
            'Kijun_sen': ichimoku.ichimoku_base_line(),
            'Senkou_Span_A': ichimoku.ichimoku_a(),
            'Senkou_Span_B': ichimoku.ichimoku_b()
        }, index=high.index)
