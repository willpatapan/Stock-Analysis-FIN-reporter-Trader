#!/usr/bin/env python3
"""
AAPL Stock Analysis and Financial Modeling Script
Analyzes Apple stock with technical indicators and predictive modeling

CHANGELOG
=========
Version History             Author          Date
@changelog   1.0.0                  WP              29-12-2025

- Initial version: Fetches AAPL stock data, computes technical indicators,
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import ta
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install yfinance pandas numpy matplotlib seaborn scikit-learn ta finnhub-python")
    exit(1)

# Optional: Finnhub for real-time data
try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    print("\n[INFO] finnhub-python not installed. Real-time data disabled.")
    print("Install with: pip install finnhub-python")
    print("Get free API key at: https://finnhub.io/register\n")


class StockAnalyzer:
    """Comprehensive stock analyzer with predictive modeling"""

    def __init__(self, ticker="AAPL", start_date=None, end_date=None, finnhub_api_key=None):
        """Initialize analyzer with ticker symbol and date range"""
        self.ticker = ticker.upper()
        # Use today as end date - yfinance will fetch up to the last trading day
        today = datetime.now()
        self.end_date = end_date or today.strftime('%Y-%m-%d')
        self.start_date = start_date or (today - timedelta(days=730)).strftime('%Y-%m-%d')
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.min_data_required = 250  # Minimum trading days needed for analysis
        
        # Initialize Finnhub client for real-time data
        self.finnhub_client = None
        if FINNHUB_AVAILABLE:
            api_key = finnhub_api_key or os.environ.get('FINNHUB_API_KEY')
            if api_key:
                try:
                    self.finnhub_client = finnhub.Client(api_key=api_key)
                    print("[✓] Finnhub real-time data enabled")
                except Exception as e:
                    print(f"[!] Finnhub initialization failed: {e}")
            else:
                print("[!] No Finnhub API key found. Set FINNHUB_API_KEY env variable or pass api_key parameter")
                print("    Get free key at: https://finnhub.io/register")

    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        print(f"Fetching {self.ticker} data from {self.start_date} to {self.end_date}...")

        # Method 1: Try Ticker object with history method
        try:
            print("Attempting to fetch data using yfinance Ticker...")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)

            if not self.data.empty:
                print(f"Successfully fetched data using Ticker method")
                # Ensure we have the required columns
                if 'Close' in self.data.columns:
                    self.data = self.data
                else:
                    raise ValueError("Missing required columns in data")
            else:
                raise ValueError("Empty data returned from Ticker method")

        except Exception as e1:
            print(f"Ticker method failed: {e1}")

            # Method 2: Try download with different parameters
            try:
                print("Attempting to fetch data using yf.download...")
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context

                self.data = yf.download(
                    self.ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                    repair=True
                )

                if not self.data.empty:
                    print(f"Successfully fetched data using download method")
                else:
                    raise ValueError("Empty data returned from download method")

            except Exception as e2:
                print(f"Download method also failed: {e2}")

                # Method 3: Try with period instead of dates
                try:
                    print("Attempting to fetch data using period parameter...")
                    stock = yf.Ticker(self.ticker)
                    self.data = stock.history(period="2y", auto_adjust=True)

                    if not self.data.empty:
                        print(f"Successfully fetched data using period method")
                    else:
                        raise ValueError("All methods failed to fetch data")

                except Exception as e3:
                    print(f"Period method also failed: {e3}")
                    raise ValueError(
                        "Unable to fetch stock data. Possible causes:\n"
                        "1. Network/firewall issues\n"
                        "2. Yahoo Finance API issues\n"
                        "3. yfinance package needs updating (try: pip install --upgrade yfinance)\n"
                        f"Last error: {e3}"
                    )

        if self.data.empty:
            raise ValueError("No data fetched. Check your date range and internet connection.")

        # Flatten multi-level columns if necessary
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = [col[0] if isinstance(col, tuple) else col for col in self.data.columns]

        print(f"Successfully fetched {len(self.data)} trading days of data")

        # Check if we have enough data for technical analysis
        if len(self.data) < self.min_data_required:
            raise ValueError(
                f"\n{'='*80}\n"
                f"INSUFFICIENT DATA ERROR\n"
                f"{'='*80}\n"
                f"Stock: {self.ticker}\n"
                f"Data points fetched: {len(self.data)}\n"
                f"Minimum required: {self.min_data_required}\n\n"
                f"This stock doesn't have enough trading history for technical analysis.\n"
                f"Possible reasons:\n"
                f"  1. Recently listed company (IPO or SPAC)\n"
                f"  2. Delisted or suspended trading\n"
                f"  3. Symbol may be incorrect\n\n"
                f"Recommendations:\n"
                f"  - Try a well-established stock like AAPL, MSFT, GOOGL, TSLA\n"
                f"  - Check if the symbol is correct\n"
                f"  - Wait until the stock has more trading history\n"
                f"{'='*80}\n"
            )

        return self.data

    def fetch_realtime_quote(self):
        """Fetch real-time quote from Finnhub API"""
        if not self.finnhub_client:
            return None
        
        try:
            quote = self.finnhub_client.quote(self.ticker)
            if quote and quote.get('c'):  # 'c' is current price
                return {
                    'current_price': quote['c'],
                    'change': quote['d'],
                    'percent_change': quote['dp'],
                    'high': quote['h'],
                    'low': quote['l'],
                    'open': quote['o'],
                    'previous_close': quote['pc'],
                    'timestamp': datetime.fromtimestamp(quote['t']).strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            print(f"[!] Error fetching real-time quote: {e}")
        
        return None

    def calculate_technical_indicators(self):
        """Calculate various technical indicators"""
        print("\nCalculating technical indicators...")

        df = self.data.copy()

        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # Average True Range (Volatility)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # On-Balance Volume
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

        # Price changes and returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']

        # Volatility (rolling standard deviation)
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

        # Trend indicator
        df['Trend'] = np.where(df['Close'] > df['SMA_50'], 1, -1)

        self.data = df.dropna()
        print(f"Technical indicators calculated. Dataset now has {len(self.data)} rows after cleaning.")

        return self.data

    def create_features(self):
        """Create features for machine learning model"""
        print("\nCreating ML features...")

        df = self.data.copy()

        # Lag features (previous days' prices)
        for i in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
            df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()

        # Target variable: predict next day's price change direction (1 for up, 0 for down)
        df['Target_Price'] = df['Close'].shift(-1)
        df['Target_Direction'] = np.where(df['Target_Price'] > df['Close'], 1, 0)
        df['Target_Return'] = (df['Target_Price'] - df['Close']) / df['Close']

        self.data = df.dropna()
        print(f"Features created. Final dataset has {len(self.data)} rows.")

        return self.data

    def train_model(self):
        """Train predictive models"""
        print("\nTraining predictive models...")

        # Select features for modeling
        feature_cols = [col for col in self.data.columns if col not in
                       ['Target_Price', 'Target_Direction', 'Target_Return', 'Dividends', 'Stock Splits']]

        X = self.data[feature_cols]
        y_price = self.data['Target_Price']
        y_direction = self.data['Target_Direction']

        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
        y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train price prediction model (Gradient Boosting)
        print("Training price prediction model...")
        price_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        price_model.fit(X_train_scaled, y_price_train)

        # Predictions
        y_pred_train = price_model.predict(X_train_scaled)
        y_pred_test = price_model.predict(X_test_scaled)

        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_price_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_price_test, y_pred_test))
        train_mae = mean_absolute_error(y_price_train, y_pred_train)
        test_mae = mean_absolute_error(y_price_test, y_pred_test)
        train_r2 = r2_score(y_price_train, y_pred_train)
        test_r2 = r2_score(y_price_test, y_pred_test)

        print(f"\nModel Performance:")
        print(f"Train RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}")
        print(f"Train MAE: ${train_mae:.2f} | Test MAE: ${test_mae:.2f}")
        print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

        # Direction accuracy
        direction_pred = np.where(y_pred_test > X_test['Close'].values, 1, 0)
        direction_accuracy = (direction_pred == y_direction_test.values).mean()
        print(f"Direction Prediction Accuracy: {direction_accuracy*100:.2f}%")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': price_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

        self.model = price_model
        self.feature_cols = feature_cols
        self.X_test = X_test
        self.y_test = y_price_test
        self.y_pred = y_pred_test

        return price_model

    def generate_signals(self):
        """Generate trading signals based on technical analysis"""
        print("\nGenerating trading signals...")

        latest = self.data.iloc[-1]
        signals = []

        # RSI signals
        if latest['RSI'] < 30:
            signals.append(("BUY", "RSI Oversold", f"RSI: {latest['RSI']:.2f}"))
        elif latest['RSI'] > 70:
            signals.append(("SELL", "RSI Overbought", f"RSI: {latest['RSI']:.2f}"))

        # MACD signals
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append(("BUY", "MACD Bullish Crossover", f"MACD: {latest['MACD']:.2f}"))
        else:
            signals.append(("SELL", "MACD Bearish", f"MACD: {latest['MACD']:.2f}"))

        # Moving Average signals
        if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
            signals.append(("BUY", "Golden Cross (Bullish Trend)", f"Price: ${latest['Close']:.2f}"))
        elif latest['Close'] < latest['SMA_50'] < latest['SMA_200']:
            signals.append(("SELL", "Death Cross (Bearish Trend)", f"Price: ${latest['Close']:.2f}"))

        # Bollinger Bands
        if latest['Close'] < latest['BB_Low']:
            signals.append(("BUY", "Price below Lower Bollinger Band", f"Price: ${latest['Close']:.2f}"))
        elif latest['Close'] > latest['BB_High']:
            signals.append(("SELL", "Price above Upper Bollinger Band", f"Price: ${latest['Close']:.2f}"))

        return signals

    def predict_future(self, days=5):
        """Predict future prices"""
        print(f"\nPredicting next {days} days...")

        predictions = []
        current_data = self.data.iloc[-1:].copy()

        for day in range(days):
            # Prepare features
            X_pred = current_data[self.feature_cols].values
            X_pred_scaled = self.scaler.transform(X_pred)

            # Predict
            pred_price = self.model.predict(X_pred_scaled)[0]
            predictions.append(pred_price)

            # Update for next iteration (simplified - in reality would recalculate all indicators)
            # This is a basic approach; more sophisticated methods would recalculate all features

        return predictions

    def visualize_analysis(self):
        """Create comprehensive visualizations"""
        print("\nGenerating visualizations...")

        fig = plt.figure(figsize=(20, 12))

        # 1. Price and Moving Averages
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        ax1.plot(self.data.index, self.data['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_200'], label='SMA 200', alpha=0.7)
        ax1.set_title(f'{self.ticker} Price with Moving Averages', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax2.set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. MACD
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', linewidth=2)
        ax3.plot(self.data.index, self.data['MACD_Signal'], label='Signal', linewidth=2)
        ax3.bar(self.data.index, self.data['MACD_Diff'], label='Histogram', alpha=0.3)
        ax3.set_title('MACD Indicator', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Bollinger Bands
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.data.index, self.data['Close'], label='Close', linewidth=2)
        ax4.plot(self.data.index, self.data['BB_High'], label='Upper Band', linestyle='--', alpha=0.7)
        ax4.plot(self.data.index, self.data['BB_Mid'], label='Middle Band', linestyle='--', alpha=0.7)
        ax4.plot(self.data.index, self.data['BB_Low'], label='Lower Band', linestyle='--', alpha=0.7)
        ax4.fill_between(self.data.index, self.data['BB_Low'], self.data['BB_High'], alpha=0.1)
        ax4.set_title('Bollinger Bands', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Volume
        ax5 = plt.subplot(3, 3, 5)
        ax5.bar(self.data.index, self.data['Volume'], alpha=0.6)
        ax5.set_title('Trading Volume', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Volume')
        ax5.grid(True, alpha=0.3)

        # 6. Daily Returns Distribution
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(self.data['Daily_Return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax6.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Daily Return')
        ax6.set_ylabel('Frequency')
        ax6.axvline(x=0, color='r', linestyle='--')
        ax6.grid(True, alpha=0.3)

        # 7. Volatility
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(self.data.index, self.data['Volatility_20'], color='orange', linewidth=2)
        ax7.set_title('20-Day Rolling Volatility', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Volatility')
        ax7.grid(True, alpha=0.3)

        # 8. Actual vs Predicted Prices (Test Set)
        ax8 = plt.subplot(3, 3, 8)
        test_dates = self.X_test.index
        ax8.plot(test_dates, self.y_test.values, label='Actual', linewidth=2)
        ax8.plot(test_dates, self.y_pred, label='Predicted', linewidth=2, alpha=0.7)
        ax8.set_title('Model Predictions (Test Set)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Price ($)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Prediction Error
        ax9 = plt.subplot(3, 3, 9)
        errors = self.y_test.values - self.y_pred
        ax9.scatter(self.y_test.values, errors, alpha=0.5)
        ax9.axhline(y=0, color='r', linestyle='--')
        ax9.set_title('Prediction Errors', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Actual Price ($)')
        ax9.set_ylabel('Error ($)')
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{self.ticker.lower()}_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as '{filename}'")
        plt.show()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print(f"{self.ticker} STOCK ANALYSIS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Show real-time quote if available
        realtime = self.fetch_realtime_quote()
        if realtime:
            print(f"\n🔴 LIVE QUOTE (Real-Time via Finnhub):")
            print(f"Current Price: ${realtime['current_price']:.2f}")
            print(f"Change: ${realtime['change']:.2f} ({realtime['percent_change']:+.2f}%)")
            print(f"Day High: ${realtime['high']:.2f}")
            print(f"Day Low: ${realtime['low']:.2f}")
            print(f"Open: ${realtime['open']:.2f}")
            print(f"Previous Close: ${realtime['previous_close']:.2f}")
            print(f"Last Updated: {realtime['timestamp']}")

        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]

        print(f"\n📊 HISTORICAL PRICE INFORMATION (End of Day):")
        print(f"Current Price: ${latest['Close']:.2f}")
        print(f"Previous Close: ${prev['Close']:.2f}")
        print(f"Daily Change: ${latest['Price_Change']:.2f} ({latest['Daily_Return']*100:.2f}%)")
        print(f"Day High: ${latest['High']:.2f}")
        print(f"Day Low: ${latest['Low']:.2f}")
        print(f"Volume: {latest['Volume']:,.0f}")

        print(f"\nTECHNICAL INDICATORS:")
        print(f"RSI (14): {latest['RSI']:.2f} - {'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}")
        print(f"MACD: {latest['MACD']:.2f}")
        print(f"MACD Signal: {latest['MACD_Signal']:.2f}")
        print(f"20-Day SMA: ${latest['SMA_20']:.2f}")
        print(f"50-Day SMA: ${latest['SMA_50']:.2f}")
        print(f"200-Day SMA: ${latest['SMA_200']:.2f}")
        print(f"Bollinger Upper: ${latest['BB_High']:.2f}")
        print(f"Bollinger Lower: ${latest['BB_Low']:.2f}")
        print(f"ATR (Volatility): {latest['ATR']:.2f}")

        print(f"\nTRADING SIGNALS:")
        signals = self.generate_signals()
        for signal_type, reason, detail in signals:
            print(f"[{signal_type}] {reason} - {detail}")

        print(f"\nSTATISTICAL SUMMARY (Last 30 Days):")
        last_30 = self.data.tail(30)
        print(f"Average Price: ${last_30['Close'].mean():.2f}")
        print(f"Price Std Dev: ${last_30['Close'].std():.2f}")
        print(f"Average Volume: {last_30['Volume'].mean():,.0f}")
        print(f"Average Daily Return: {last_30['Daily_Return'].mean()*100:.2f}%")
        print(f"Volatility: {last_30['Daily_Return'].std()*100:.2f}%")

        # Future predictions
        future_prices = self.predict_future(days=5)
        print(f"\nPRICE PREDICTIONS (Next 5 Days):")
        for i, price in enumerate(future_prices, 1):
            change = ((price - latest['Close']) / latest['Close']) * 100
            print(f"Day {i}: ${price:.2f} ({change:+.2f}%)")

        print(f"\nRISK ASSESSMENT:")
        volatility = last_30['Daily_Return'].std() * 100
        if volatility > 3:
            risk_level = "HIGH"
        elif volatility > 1.5:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        print(f"Risk Level: {risk_level} (Volatility: {volatility:.2f}%)")

        print("\n" + "="*80)
        print("DISCLAIMER: This analysis is for educational purposes only.")
        print("Not financial advice. Always do your own research before investing.")
        print("="*80)


def main():
    """Main execution function"""
    import sys

    # Parse command line arguments
    ticker = "AAPL"  # Default ticker (Apple Inc.)
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()

    print("="*80)
    print(f"{ticker} STOCK ANALYZER - Financial Modeling & Technical Analysis")
    print("="*80)
    print(f"Tip: Run with any ticker symbol - python3 {sys.argv[0]} MSFT")
    print("="*80)

    # Initialize analyzer (default: last 2 years)
    analyzer = StockAnalyzer(ticker=ticker)

    try:
        # Fetch data
        analyzer.fetch_data()

        # Calculate indicators
        analyzer.calculate_technical_indicators()

        # Create features
        analyzer.create_features()

        # Train model
        analyzer.train_model()

        # Generate report
        analyzer.generate_report()

        # Create visualizations
        analyzer.visualize_analysis()

        print(f"\nAnalysis complete! Check '{ticker.lower()}_analysis.png' for visualizations.")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
