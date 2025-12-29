#!/usr/bin/env python3
"""
Automated Trading Bot Simulation
Simulates algorithmic trading with configurable parameters and risk management

CHANGELOG
=========
Version History             Author          Date
@changelog   1.0.0                  WP              29-12-2025

- Initial version: Automated trading bot simulation with technical analysis,
  position sizing, risk management, and performance tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    import ta
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install yfinance pandas numpy matplotlib seaborn ta")
    exit(1)


class TradingBot:
    """Automated trading bot with simulation capabilities"""

    def __init__(self, ticker="AAPL", initial_capital=10000,
                 max_position_size=0.2, min_position_size=0.05,
                 stop_loss_pct=0.05, take_profit_pct=0.10,
                 start_date=None, end_date=None):
        """
        Initialize trading bot with configuration

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        initial_capital : float
            Starting capital in dollars
        max_position_size : float
            Maximum position size as percentage of portfolio (0.0-1.0)
        min_position_size : float
            Minimum position size as percentage of portfolio (0.0-1.0)
        stop_loss_pct : float
            Stop loss percentage (0.0-1.0)
        take_profit_pct : float
            Take profit percentage (0.0-1.0)
        start_date : str
            Start date for simulation (YYYY-MM-DD)
        end_date : str
            End date for simulation (YYYY-MM-DD)
        """
        self.ticker = ticker.upper()
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Date range
        today = datetime.now()
        self.end_date = end_date or today.strftime('%Y-%m-%d')
        self.start_date = start_date or (today - timedelta(days=365)).strftime('%Y-%m-%d')

        # Portfolio state
        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = initial_capital

        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.data = None

    def fetch_data(self):
        """Fetch historical stock data"""
        print(f"Fetching {self.ticker} data from {self.start_date} to {self.end_date}...")

        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)

            if self.data.empty:
                raise ValueError("No data fetched")

            print(f"Successfully fetched {len(self.data)} trading days")
            return self.data

        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def calculate_indicators(self):
        """Calculate technical indicators for trading signals"""
        print("Calculating technical indicators...")

        df = self.data.copy()

        # Moving Averages
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
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

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # ATR for volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Price momentum
        df['ROC'] = ta.momentum.roc(df['Close'], window=10)
        df['Daily_Return'] = df['Close'].pct_change()

        self.data = df.dropna()
        print(f"Indicators calculated. {len(self.data)} rows ready for trading.")

        return self.data

    def generate_signals(self, row):
        """
        Generate trading signals based on multiple indicators

        Returns:
        --------
        signal : int
            1 for BUY, -1 for SELL, 0 for HOLD
        strength : float
            Signal strength (0.0-1.0)
        """
        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        # 1. RSI Signal
        if row['RSI'] < 30:
            buy_signals += 1
        elif row['RSI'] > 70:
            sell_signals += 1
        total_signals += 1

        # 2. MACD Signal
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Diff'] > 0:
            buy_signals += 1
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Diff'] < 0:
            sell_signals += 1
        total_signals += 1

        # 3. Moving Average Crossover
        if row['SMA_10'] > row['SMA_20'] and row['Close'] > row['SMA_50']:
            buy_signals += 1
        elif row['SMA_10'] < row['SMA_20'] and row['Close'] < row['SMA_50']:
            sell_signals += 1
        total_signals += 1

        # 4. Bollinger Bands
        if row['Close'] < row['BB_Low']:
            buy_signals += 1
        elif row['Close'] > row['BB_High']:
            sell_signals += 1
        total_signals += 1

        # 5. Stochastic Oscillator
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']:
            buy_signals += 1
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']:
            sell_signals += 1
        total_signals += 1

        # 6. Volume confirmation
        volume_confirmed = row['Volume_Ratio'] > 1.2

        # Calculate signal strength
        buy_strength = buy_signals / total_signals
        sell_strength = sell_signals / total_signals

        # Determine final signal (require at least 60% consensus)
        if buy_strength >= 0.6 and volume_confirmed:
            return 1, buy_strength
        elif sell_strength >= 0.6:
            return -1, sell_strength
        else:
            return 0, max(buy_strength, sell_strength)

    def calculate_position_size(self, price, signal_strength):
        """
        Calculate position size based on signal strength and risk management

        Parameters:
        -----------
        price : float
            Current stock price
        signal_strength : float
            Strength of trading signal (0.0-1.0)

        Returns:
        --------
        shares : int
            Number of shares to trade
        """
        # Scale position size based on signal strength
        position_pct = self.min_position_size + (
            (self.max_position_size - self.min_position_size) * signal_strength
        )

        # Calculate dollar amount to invest
        investment_amount = self.portfolio_value * position_pct

        # Don't invest more cash than available
        investment_amount = min(investment_amount, self.cash)

        # Calculate shares (round down to avoid overspending)
        shares = int(investment_amount / price)

        return shares

    def execute_trade(self, date, price, signal, signal_strength, reason):
        """Execute a trade and update portfolio"""

        if signal == 1:  # BUY
            shares_to_buy = self.calculate_position_size(price, signal_strength)

            if shares_to_buy > 0 and self.cash >= shares_to_buy * price:
                cost = shares_to_buy * price
                self.shares += shares_to_buy
                self.cash -= cost

                trade = {
                    'Date': date,
                    'Action': 'BUY',
                    'Price': price,
                    'Shares': shares_to_buy,
                    'Cost': cost,
                    'Cash_After': self.cash,
                    'Shares_After': self.shares,
                    'Signal_Strength': signal_strength,
                    'Reason': reason
                }
                self.trades.append(trade)

                return True

        elif signal == -1:  # SELL
            if self.shares > 0:
                # Sell all shares
                revenue = self.shares * price
                shares_sold = self.shares

                self.cash += revenue
                self.shares = 0

                trade = {
                    'Date': date,
                    'Action': 'SELL',
                    'Price': price,
                    'Shares': shares_sold,
                    'Revenue': revenue,
                    'Cash_After': self.cash,
                    'Shares_After': self.shares,
                    'Signal_Strength': signal_strength,
                    'Reason': reason
                }
                self.trades.append(trade)

                return True

        return False

    def check_stop_loss_take_profit(self, current_price):
        """Check if stop loss or take profit should be triggered"""

        if self.shares == 0:
            return 0, 0, None

        # Get entry price (last buy trade)
        for trade in reversed(self.trades):
            if trade['Action'] == 'BUY':
                entry_price = trade['Price']
                break
        else:
            return 0, 0, None

        # Calculate profit/loss percentage
        pnl_pct = (current_price - entry_price) / entry_price

        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return -1, 1.0, f"Stop Loss triggered ({pnl_pct*100:.2f}%)"

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return -1, 1.0, f"Take Profit triggered ({pnl_pct*100:.2f}%)"

        return 0, 0, None

    def run_simulation(self):
        """Run the trading bot simulation"""
        print("\n" + "="*80)
        print(f"STARTING TRADING BOT SIMULATION - {self.ticker}")
        print("="*80)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Position Size: {self.min_position_size*100:.1f}% - {self.max_position_size*100:.1f}%")
        print(f"Stop Loss: {self.stop_loss_pct*100:.1f}%")
        print(f"Take Profit: {self.take_profit_pct*100:.1f}%")
        print("="*80 + "\n")

        for idx, row in self.data.iterrows():
            price = row['Close']

            # Update portfolio value
            self.portfolio_value = self.cash + (self.shares * price)

            # Record portfolio state
            self.portfolio_history.append({
                'Date': idx,
                'Price': price,
                'Cash': self.cash,
                'Shares': self.shares,
                'Portfolio_Value': self.portfolio_value
            })

            # Check stop loss / take profit first
            sl_tp_signal, sl_tp_strength, sl_tp_reason = self.check_stop_loss_take_profit(price)

            if sl_tp_signal != 0:
                self.execute_trade(idx, price, sl_tp_signal, sl_tp_strength, sl_tp_reason)
                continue

            # Generate trading signals
            signal, strength = self.generate_signals(row)

            # Execute trades based on signals
            if signal == 1 and self.shares == 0:
                # Only buy if we don't have a position
                reason = f"BUY Signal (Strength: {strength:.2f})"
                self.execute_trade(idx, price, signal, strength, reason)

            elif signal == -1 and self.shares > 0:
                # Only sell if we have a position
                reason = f"SELL Signal (Strength: {strength:.2f})"
                self.execute_trade(idx, price, signal, strength, reason)

        # Final portfolio value
        final_price = self.data.iloc[-1]['Close']
        final_value = self.cash + (self.shares * final_price)

        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: ${final_value - self.initial_capital:,.2f} ({((final_value/self.initial_capital)-1)*100:.2f}%)")
        print(f"Total Trades: {len(self.trades)}")
        print("="*80 + "\n")

    def calculate_metrics(self):
        """Calculate performance metrics"""

        if len(self.trades) == 0:
            print("No trades executed during simulation.")
            return None

        # Convert to DataFrames
        trades_df = pd.DataFrame(self.trades)
        portfolio_df = pd.DataFrame(self.portfolio_history)

        # Calculate returns
        portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()

        # Buy & Hold comparison
        initial_price = self.data.iloc[0]['Close']
        final_price = self.data.iloc[-1]['Close']
        buy_hold_return = ((final_price / initial_price) - 1) * 100

        # Bot performance
        bot_return = ((portfolio_df.iloc[-1]['Portfolio_Value'] / self.initial_capital) - 1) * 100

        # Win rate
        buy_trades = trades_df[trades_df['Action'] == 'BUY']
        sell_trades = trades_df[trades_df['Action'] == 'SELL']

        if len(sell_trades) > 0 and len(buy_trades) > 0:
            wins = 0
            losses = 0

            for sell_idx, sell_trade in sell_trades.iterrows():
                # Find corresponding buy trade
                buy_trade = buy_trades[buy_trades['Date'] < sell_trade['Date']].iloc[-1]
                profit = sell_trade['Price'] - buy_trade['Price']

                if profit > 0:
                    wins += 1
                else:
                    losses += 1

            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        else:
            win_rate = 0

        # Sharpe Ratio (annualized)
        if portfolio_df['Returns'].std() > 0:
            sharpe_ratio = (portfolio_df['Returns'].mean() / portfolio_df['Returns'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum Drawdown
        portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Peak']) / portfolio_df['Peak']
        max_drawdown = portfolio_df['Drawdown'].min() * 100

        metrics = {
            'Total Trades': len(self.trades),
            'Buy Trades': len(buy_trades),
            'Sell Trades': len(sell_trades),
            'Win Rate': win_rate,
            'Bot Return': bot_return,
            'Buy & Hold Return': buy_hold_return,
            'Excess Return': bot_return - buy_hold_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Final Portfolio Value': portfolio_df.iloc[-1]['Portfolio_Value']
        }

        return metrics, trades_df, portfolio_df

    def print_report(self):
        """Print comprehensive trading report"""

        metrics, trades_df, portfolio_df = self.calculate_metrics()

        if metrics is None:
            return

        print("\n" + "="*80)
        print(f"TRADING BOT PERFORMANCE REPORT - {self.ticker}")
        print("="*80)

        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${metrics['Final Portfolio Value']:,.2f}")
        print(f"Total Profit/Loss: ${metrics['Final Portfolio Value'] - self.initial_capital:,.2f}")
        print(f"Return: {metrics['Bot Return']:.2f}%")

        print(f"\nCOMPARISON:")
        print(f"Buy & Hold Return: {metrics['Buy & Hold Return']:.2f}%")
        print(f"Excess Return: {metrics['Excess Return']:.2f}%")
        print(f"Bot {'OUTPERFORMED' if metrics['Excess Return'] > 0 else 'UNDERPERFORMED'} buy-and-hold strategy")

        print(f"\nTRADING STATISTICS:")
        print(f"Total Trades: {metrics['Total Trades']}")
        print(f"Buy Trades: {metrics['Buy Trades']}")
        print(f"Sell Trades: {metrics['Sell Trades']}")
        print(f"Win Rate: {metrics['Win Rate']:.2f}%")

        print(f"\nRISK METRICS:")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['Max Drawdown']:.2f}%")

        print(f"\nRECENT TRADES:")
        if len(trades_df) > 0:
            print(trades_df.tail(10).to_string(index=False))

        print("\n" + "="*80)
        print("DISCLAIMER: This is a simulation for educational purposes only.")
        print("Past performance does not guarantee future results.")
        print("="*80 + "\n")

    def visualize_results(self):
        """Create visualization of trading results"""

        metrics, trades_df, portfolio_df = self.calculate_metrics()

        if metrics is None:
            return

        print("Generating visualizations...")

        fig = plt.figure(figsize=(20, 12))

        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(portfolio_df['Date'], portfolio_df['Portfolio_Value'],
                label='Bot Portfolio', linewidth=2, color='blue')

        # Buy & Hold comparison
        initial_price = self.data.iloc[0]['Close']
        buy_hold_values = (self.data['Close'] / initial_price) * self.initial_capital
        ax1.plot(self.data.index, buy_hold_values,
                label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)

        ax1.set_title(f'{self.ticker} - Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Stock Price with Buy/Sell Signals
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.data.index, self.data['Close'], label='Price', linewidth=2, color='black')

        # Mark buy trades
        buy_trades = trades_df[trades_df['Action'] == 'BUY']
        if len(buy_trades) > 0:
            ax2.scatter(buy_trades['Date'], buy_trades['Price'],
                       color='green', marker='^', s=100, label='Buy', zorder=5)

        # Mark sell trades
        sell_trades = trades_df[trades_df['Action'] == 'SELL']
        if len(sell_trades) > 0:
            ax2.scatter(sell_trades['Date'], sell_trades['Price'],
                       color='red', marker='v', s=100, label='Sell', zorder=5)

        ax2.set_title('Price Chart with Trading Signals', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative Returns
        ax3 = plt.subplot(3, 2, 3)
        portfolio_df['Cumulative_Return'] = ((portfolio_df['Portfolio_Value'] / self.initial_capital) - 1) * 100
        buy_hold_returns = ((buy_hold_values / self.initial_capital) - 1) * 100

        ax3.plot(portfolio_df['Date'], portfolio_df['Cumulative_Return'],
                label='Bot', linewidth=2, color='blue')
        ax3.plot(self.data.index, buy_hold_returns,
                label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Drawdown
        ax4 = plt.subplot(3, 2, 4)
        ax4.fill_between(portfolio_df['Date'], portfolio_df['Drawdown'] * 100, 0,
                        color='red', alpha=0.3)
        ax4.plot(portfolio_df['Date'], portfolio_df['Drawdown'] * 100,
                color='red', linewidth=2)
        ax4.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Trade Distribution
        ax5 = plt.subplot(3, 2, 5)
        if len(trades_df) > 0:
            trade_counts = trades_df['Action'].value_counts()
            colors = ['green' if x == 'BUY' else 'red' for x in trade_counts.index]
            ax5.bar(trade_counts.index, trade_counts.values, color=colors, alpha=0.7)
            ax5.set_title('Trade Distribution', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Action')
            ax5.set_ylabel('Count')
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. Performance Metrics Summary
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')

        summary_text = f"""
        PERFORMANCE SUMMARY
        ═══════════════════════════════════════

        Initial Capital:        ${self.initial_capital:,.2f}
        Final Value:            ${metrics['Final Portfolio Value']:,.2f}
        Total Return:           {metrics['Bot Return']:.2f}%

        Buy & Hold Return:      {metrics['Buy & Hold Return']:.2f}%
        Excess Return:          {metrics['Excess Return']:.2f}%

        Total Trades:           {metrics['Total Trades']}
        Win Rate:               {metrics['Win Rate']:.2f}%

        Sharpe Ratio:           {metrics['Sharpe Ratio']:.2f}
        Max Drawdown:           {metrics['Max Drawdown']:.2f}%

        Position Size:          {self.min_position_size*100:.1f}% - {self.max_position_size*100:.1f}%
        Stop Loss:              {self.stop_loss_pct*100:.1f}%
        Take Profit:            {self.take_profit_pct*100:.1f}%
        """

        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        filename = f'{self.ticker.lower()}_trading_bot_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as '{filename}'")
        plt.show()


def main():
    """Main execution function"""

    print("="*80)
    print("AUTOMATED TRADING BOT SIMULATION")
    print("="*80)

    # Configure trading bot parameters
    config = {
        'ticker': 'AAPL',              # Stock to trade
        'initial_capital': 10000,       # Starting capital
        'max_position_size': 0.25,      # Maximum 25% of portfolio per trade
        'min_position_size': 0.10,      # Minimum 10% of portfolio per trade
        'stop_loss_pct': 0.05,          # 5% stop loss
        'take_profit_pct': 0.10,        # 10% take profit
        'start_date': None,             # Will default to 1 year ago
        'end_date': None                # Will default to today
    }

    print("\nBot Configuration:")
    print(f"  Ticker: {config['ticker']}")
    print(f"  Initial Capital: ${config['initial_capital']:,.2f}")
    print(f"  Position Size: {config['min_position_size']*100:.0f}% - {config['max_position_size']*100:.0f}%")
    print(f"  Stop Loss: {config['stop_loss_pct']*100:.0f}%")
    print(f"  Take Profit: {config['take_profit_pct']*100:.0f}%")
    print("\n" + "="*80 + "\n")

    try:
        # Initialize bot
        bot = TradingBot(**config)

        # Fetch data
        bot.fetch_data()

        # Calculate indicators
        bot.calculate_indicators()

        # Run simulation
        bot.run_simulation()

        # Generate report
        bot.print_report()

        # Visualize results
        bot.visualize_results()

        print(f"\nSimulation complete! Check '{config['ticker'].lower()}_trading_bot_results.png' for visualizations.")

    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
