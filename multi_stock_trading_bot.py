#!/usr/bin/env python3
"""
Multi-Stock Automated Trading Bot Simulation
Runs trading bot simulation across top 100 stocks and compares performance

CHANGELOG
=========
Version History             Author          Date
@changelog   1.0.0                  WP              29-12-2025

- Initial version: Multi-stock trading bot simulation with performance ranking
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install yfinance pandas numpy matplotlib seaborn ta")
    exit(1)


# Top liquid US stocks for trading simulation
SP100_TICKERS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Blue Chips
    'JPM', 'V', 'WMT', 'PG', 'JNJ', 'UNH', 'MA', 'HD',
    # High Growth
    'NFLX', 'DIS', 'PYPL', 'ADBE', 'CRM', 'INTC', 'CSCO',
    # Finance
    'BAC', 'GS', 'MS', 'C', 'WFC',
    # Consumer
    'KO', 'PEP', 'NKE', 'MCD', 'SBUX',
    # Healthcare
    'PFE', 'ABBV', 'TMO', 'MRK', 'LLY'
]


class TradingBot:
    """Automated trading bot - same as original"""

    def __init__(self, ticker, initial_capital=10000,
                 max_position_size=0.2, min_position_size=0.05,
                 stop_loss_pct=0.05, take_profit_pct=0.10,
                 start_date=None, end_date=None, verbose=False):

        self.ticker = ticker.upper()
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.verbose = verbose

        today = datetime.now()
        self.end_date = end_date or today.strftime('%Y-%m-%d')
        self.start_date = start_date or (today - timedelta(days=365)).strftime('%Y-%m-%d')

        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = initial_capital
        self.trades = []
        self.portfolio_history = []
        self.data = None

    def fetch_data(self):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)

            if self.data.empty:
                return False

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error fetching {self.ticker}: {e}")
            return False

    def calculate_indicators(self):
        """Calculate technical indicators"""
        try:
            df = self.data.copy()

            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()

            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            df['BB_Mid'] = bollinger.bollinger_mavg()

            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()

            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['ROC'] = ta.momentum.roc(df['Close'], window=10)
            df['Daily_Return'] = df['Close'].pct_change()

            self.data = df.dropna()
            return True

        except Exception as e:
            if self.verbose:
                print(f"Error calculating indicators for {self.ticker}: {e}")
            return False

    def generate_signals(self, row):
        """Generate trading signals"""
        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        if row['RSI'] < 30:
            buy_signals += 1
        elif row['RSI'] > 70:
            sell_signals += 1
        total_signals += 1

        if row['MACD'] > row['MACD_Signal'] and row['MACD_Diff'] > 0:
            buy_signals += 1
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Diff'] < 0:
            sell_signals += 1
        total_signals += 1

        if row['SMA_10'] > row['SMA_20'] and row['Close'] > row['SMA_50']:
            buy_signals += 1
        elif row['SMA_10'] < row['SMA_20'] and row['Close'] < row['SMA_50']:
            sell_signals += 1
        total_signals += 1

        if row['Close'] < row['BB_Low']:
            buy_signals += 1
        elif row['Close'] > row['BB_High']:
            sell_signals += 1
        total_signals += 1

        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']:
            buy_signals += 1
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']:
            sell_signals += 1
        total_signals += 1

        volume_confirmed = row['Volume_Ratio'] > 1.2

        buy_strength = buy_signals / total_signals
        sell_strength = sell_signals / total_signals

        if buy_strength >= 0.6 and volume_confirmed:
            return 1, buy_strength
        elif sell_strength >= 0.6:
            return -1, sell_strength
        else:
            return 0, max(buy_strength, sell_strength)

    def calculate_position_size(self, price, signal_strength):
        """Calculate position size based on signal strength"""
        position_pct = self.min_position_size + (
            (self.max_position_size - self.min_position_size) * signal_strength
        )
        investment_amount = self.portfolio_value * position_pct
        investment_amount = min(investment_amount, self.cash)
        shares = int(investment_amount / price)
        return shares

    def execute_trade(self, date, price, signal, signal_strength, reason):
        """Execute a trade"""
        if signal == 1:
            shares_to_buy = self.calculate_position_size(price, signal_strength)
            if shares_to_buy > 0 and self.cash >= shares_to_buy * price:
                cost = shares_to_buy * price
                self.shares += shares_to_buy
                self.cash -= cost
                self.trades.append({
                    'Date': date, 'Action': 'BUY', 'Price': price,
                    'Shares': shares_to_buy, 'Cost': cost,
                    'Cash_After': self.cash, 'Shares_After': self.shares,
                    'Signal_Strength': signal_strength, 'Reason': reason
                })
                return True

        elif signal == -1:
            if self.shares > 0:
                revenue = self.shares * price
                shares_sold = self.shares
                self.cash += revenue
                self.shares = 0
                self.trades.append({
                    'Date': date, 'Action': 'SELL', 'Price': price,
                    'Shares': shares_sold, 'Revenue': revenue,
                    'Cash_After': self.cash, 'Shares_After': self.shares,
                    'Signal_Strength': signal_strength, 'Reason': reason
                })
                return True

        return False

    def check_stop_loss_take_profit(self, current_price):
        """Check stop loss / take profit"""
        if self.shares == 0:
            return 0, 0, None

        for trade in reversed(self.trades):
            if trade['Action'] == 'BUY':
                entry_price = trade['Price']
                break
        else:
            return 0, 0, None

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.stop_loss_pct:
            return -1, 1.0, f"Stop Loss ({pnl_pct*100:.2f}%)"
        if pnl_pct >= self.take_profit_pct:
            return -1, 1.0, f"Take Profit ({pnl_pct*100:.2f}%)"

        return 0, 0, None

    def run_simulation(self):
        """Run trading simulation"""
        for idx, row in self.data.iterrows():
            price = row['Close']
            self.portfolio_value = self.cash + (self.shares * price)

            self.portfolio_history.append({
                'Date': idx, 'Price': price, 'Cash': self.cash,
                'Shares': self.shares, 'Portfolio_Value': self.portfolio_value
            })

            sl_tp_signal, sl_tp_strength, sl_tp_reason = self.check_stop_loss_take_profit(price)

            if sl_tp_signal != 0:
                self.execute_trade(idx, price, sl_tp_signal, sl_tp_strength, sl_tp_reason)
                continue

            signal, strength = self.generate_signals(row)

            if signal == 1 and self.shares == 0:
                self.execute_trade(idx, price, signal, strength, f"BUY Signal ({strength:.2f})")
            elif signal == -1 and self.shares > 0:
                self.execute_trade(idx, price, signal, strength, f"SELL Signal ({strength:.2f})")

    def get_metrics(self):
        """Calculate and return performance metrics"""
        if len(self.portfolio_history) == 0:
            return None

        portfolio_df = pd.DataFrame(self.portfolio_history)
        final_value = portfolio_df.iloc[-1]['Portfolio_Value']

        # Buy & Hold comparison
        initial_price = self.data.iloc[0]['Close']
        final_price = self.data.iloc[-1]['Close']
        buy_hold_return = ((final_price / initial_price) - 1) * 100
        bot_return = ((final_value / self.initial_capital) - 1) * 100

        # Win rate
        trades_df = pd.DataFrame(self.trades)
        buy_trades = trades_df[trades_df['Action'] == 'BUY']
        sell_trades = trades_df[trades_df['Action'] == 'SELL']

        wins = 0
        losses = 0
        if len(sell_trades) > 0 and len(buy_trades) > 0:
            for _, sell_trade in sell_trades.iterrows():
                buy_trade = buy_trades[buy_trades['Date'] < sell_trade['Date']].iloc[-1]
                profit = sell_trade['Price'] - buy_trade['Price']
                if profit > 0:
                    wins += 1
                else:
                    losses += 1

        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0

        # Sharpe Ratio
        portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
        sharpe_ratio = 0
        if portfolio_df['Returns'].std() > 0:
            sharpe_ratio = (portfolio_df['Returns'].mean() / portfolio_df['Returns'].std()) * np.sqrt(252)

        # Max Drawdown
        portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] - portfolio_df['Peak']) / portfolio_df['Peak']
        max_drawdown = portfolio_df['Drawdown'].min() * 100

        return {
            'Ticker': self.ticker,
            'Total Trades': len(self.trades),
            'Win Rate': win_rate,
            'Bot Return (%)': bot_return,
            'Buy & Hold Return (%)': buy_hold_return,
            'Excess Return (%)': bot_return - buy_hold_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Final Value': final_value,
            'Profit/Loss': final_value - self.initial_capital
        }


def simulate_single_stock(ticker, config):
    """Simulate trading for a single stock"""
    try:
        bot = TradingBot(ticker, **config)

        if not bot.fetch_data():
            return None

        if len(bot.data) < 60:  # Need at least 60 days
            return None

        if not bot.calculate_indicators():
            return None

        bot.run_simulation()
        metrics = bot.get_metrics()

        return metrics

    except Exception as e:
        return None


def run_multi_stock_simulation(tickers, config, max_workers=10):
    """Run simulation across multiple stocks in parallel"""

    print("="*80)
    print(f"MULTI-STOCK TRADING BOT SIMULATION")
    print("="*80)
    print(f"Testing {len(tickers)} stocks")
    print(f"Initial Capital per stock: ${config['initial_capital']:,.2f}")
    print(f"Position Size: {config['min_position_size']*100:.0f}% - {config['max_position_size']*100:.0f}%")
    print(f"Stop Loss: {config['stop_loss_pct']*100:.0f}% | Take Profit: {config['take_profit_pct']*100:.0f}%")
    print("="*80 + "\n")

    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(simulate_single_stock, ticker, config): ticker
            for ticker in tickers
        }

        completed = 0
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1

            try:
                metrics = future.result()
                if metrics:
                    results.append(metrics)
                    print(f"[{completed}/{len(tickers)}] {ticker}: Bot Return = {metrics['Bot Return (%)']:+.2f}% | Trades = {metrics['Total Trades']}")
                else:
                    print(f"[{completed}/{len(tickers)}] {ticker}: Skipped (insufficient data)")
            except Exception as e:
                print(f"[{completed}/{len(tickers)}] {ticker}: Error - {e}")

    print("\n" + "="*80)
    print(f"Simulation Complete: {len(results)}/{len(tickers)} stocks processed successfully")
    print("="*80 + "\n")

    return pd.DataFrame(results)


def generate_summary_report(results_df):
    """Generate comprehensive summary report"""

    print("\n" + "="*80)
    print("MULTI-STOCK PERFORMANCE SUMMARY")
    print("="*80)

    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Stocks Analyzed: {len(results_df)}")
    print(f"Average Bot Return: {results_df['Bot Return (%)'].mean():.2f}%")
    print(f"Average Buy & Hold Return: {results_df['Buy & Hold Return (%)'].mean():.2f}%")
    print(f"Average Excess Return: {results_df['Excess Return (%)'].mean():.2f}%")
    print(f"Average Win Rate: {results_df['Win Rate'].mean():.2f}%")
    print(f"Average Sharpe Ratio: {results_df['Sharpe Ratio'].mean():.2f}")

    # Bot vs Buy & Hold
    outperformed = len(results_df[results_df['Excess Return (%)'] > 0])
    print(f"\nBot outperformed Buy & Hold: {outperformed}/{len(results_df)} stocks ({outperformed/len(results_df)*100:.1f}%)")

    # Top 10 performers
    print(f"\nTOP 10 BEST PERFORMERS (by Bot Return):")
    top10 = results_df.nlargest(10, 'Bot Return (%)')
    print(top10[['Ticker', 'Bot Return (%)', 'Buy & Hold Return (%)', 'Excess Return (%)', 'Win Rate']].to_string(index=False))

    # Bottom 10 performers
    print(f"\nTOP 10 WORST PERFORMERS (by Bot Return):")
    bottom10 = results_df.nsmallest(10, 'Bot Return (%)')
    print(bottom10[['Ticker', 'Bot Return (%)', 'Buy & Hold Return (%)', 'Excess Return (%)', 'Win Rate']].to_string(index=False))

    # Best excess returns (vs buy & hold)
    print(f"\nTOP 10 STOCKS WHERE BOT BEAT BUY & HOLD:")
    best_excess = results_df.nlargest(10, 'Excess Return (%)')
    print(best_excess[['Ticker', 'Bot Return (%)', 'Buy & Hold Return (%)', 'Excess Return (%)']].to_string(index=False))

    print("\n" + "="*80)


def visualize_multi_stock_results(results_df, config):
    """Create Goldman Sachs-quality visualizations for multi-stock results"""

    print("\nGenerating professional-grade visualizations...")

    # Set Goldman Sachs color palette
    GS_BLUE = '#0033A0'
    GS_LIGHT_BLUE = '#5B9BD5'
    GS_GREEN = '#70AD47'
    GS_RED = '#C00000'
    GS_GOLD = '#FFC000'
    GS_GRAY = '#7F7F7F'

    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(24, 16), facecolor='white')
    fig.suptitle('Multi-Stock Trading Strategy Performance Analysis',
                 fontsize=20, fontweight='bold', y=0.995, color=GS_BLUE)

    # 1. Individual Stock Performance - Detailed Comparison
    ax1 = plt.subplot(4, 3, 1)
    results_sorted = results_df.sort_values('Bot Return (%)', ascending=True)
    y_pos = np.arange(len(results_sorted))
    colors = [GS_GREEN if x > 0 else GS_RED for x in results_sorted['Bot Return (%)']]
    ax1.barh(y_pos, results_sorted['Bot Return (%)'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos[::max(1, len(y_pos)//10)])  # Show every nth label
    ax1.set_yticklabels(results_sorted['Ticker'].iloc[::max(1, len(y_pos)//10)], fontsize=8)
    ax1.set_xlabel('Return (%)', fontweight='bold')
    ax1.set_title('All Stocks: Bot Return Ranking', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.grid(True, alpha=0.2, axis='x')
    ax1.set_facecolor('#F8F9FA')

    # 2. Bot vs Buy & Hold with Stock Labels
    ax2 = plt.subplot(4, 3, 2)
    scatter_colors = [GS_GREEN if x > y else GS_RED
                     for x, y in zip(results_df['Bot Return (%)'], results_df['Buy & Hold Return (%)'])]
    ax2.scatter(results_df['Buy & Hold Return (%)'], results_df['Bot Return (%)'],
               c=scatter_colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

    # Add stock labels for top/bottom performers
    top5 = results_df.nlargest(3, 'Excess Return (%)')
    bottom5 = results_df.nsmallest(3, 'Excess Return (%)')
    for _, row in pd.concat([top5, bottom5]).iterrows():
        ax2.annotate(row['Ticker'], (row['Buy & Hold Return (%)'], row['Bot Return (%)']),
                    fontsize=7, alpha=0.8, fontweight='bold')

    min_val = min(results_df['Buy & Hold Return (%)'].min(), results_df['Bot Return (%)'].min())
    max_val = max(results_df['Buy & Hold Return (%)'].max(), results_df['Bot Return (%)'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], color=GS_GRAY, linestyle='--',
            linewidth=2, label='Equal Performance', alpha=0.7)
    ax2.set_title('Bot vs Buy & Hold Strategy', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax2.set_xlabel('Buy & Hold Return (%)', fontweight='bold')
    ax2.set_ylabel('Bot Return (%)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#F8F9FA')

    # 3. Excess Returns Waterfall Chart
    ax3 = plt.subplot(4, 3, 3)
    excess_sorted = results_df.nlargest(15, 'Excess Return (%)')
    colors_excess = [GS_GREEN if x > 0 else GS_RED for x in excess_sorted['Excess Return (%)']]
    bars = ax3.barh(range(len(excess_sorted)), excess_sorted['Excess Return (%)'],
                    color=colors_excess, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(excess_sorted)))
    ax3.set_yticklabels(excess_sorted['Ticker'], fontsize=9)
    ax3.set_xlabel('Excess Return vs Buy & Hold (%)', fontweight='bold')
    ax3.set_title('Top 15: Strategy Alpha Generation', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax3.grid(True, alpha=0.2, axis='x')
    ax3.invert_yaxis()
    ax3.set_facecolor('#F8F9FA')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, excess_sorted['Excess Return (%)'])):
        ax3.text(val, i, f' {val:+.1f}%', va='center', ha='left' if val > 0 else 'right',
                fontsize=7, fontweight='bold')

    # 4. Sector/Category Analysis
    ax4 = plt.subplot(4, 3, 4)
    results_df['Category'] = results_df['Ticker'].apply(lambda x:
        'Tech' if x in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO']
        else 'Finance' if x in ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC']
        else 'Healthcare' if x in ['JNJ', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY', 'UNH']
        else 'Consumer')

    category_perf = results_df.groupby('Category')['Bot Return (%)'].mean().sort_values()
    colors_cat = [GS_BLUE, GS_LIGHT_BLUE, GS_GOLD, GS_GREEN][:len(category_perf)]
    ax4.barh(category_perf.index, category_perf.values, color=colors_cat, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Average Bot Return (%)', fontweight='bold')
    ax4.set_title('Performance by Sector', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax4.grid(True, alpha=0.2, axis='x')
    ax4.set_facecolor('#F8F9FA')
    for i, v in enumerate(category_perf.values):
        ax4.text(v, i, f'  {v:.2f}%', va='center', fontsize=9, fontweight='bold')

    # 5. Risk-Return Profile
    ax5 = plt.subplot(4, 3, 5)
    scatter = ax5.scatter(results_df['Max Drawdown (%)'], results_df['Bot Return (%)'],
                         c=results_df['Sharpe Ratio'], cmap='RdYlGn', s=120,
                         alpha=0.7, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Sharpe Ratio', fontweight='bold', fontsize=9)
    ax5.set_xlabel('Max Drawdown (%)', fontweight='bold')
    ax5.set_ylabel('Bot Return (%)', fontweight='bold')
    ax5.set_title('Risk-Adjusted Performance Map', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax5.grid(True, alpha=0.2)
    ax5.set_facecolor('#F8F9FA')

    # Add quadrant lines
    ax5.axhline(y=results_df['Bot Return (%)'].median(), color=GS_GRAY, linestyle='--', alpha=0.5)
    ax5.axvline(x=results_df['Max Drawdown (%)'].median(), color=GS_GRAY, linestyle='--', alpha=0.5)

    # 6. Win Rate vs Returns
    ax6 = plt.subplot(4, 3, 6)
    scatter2 = ax6.scatter(results_df['Win Rate'], results_df['Bot Return (%)'],
                          s=results_df['Total Trades']*3, alpha=0.6, c=results_df['Sharpe Ratio'],
                          cmap='viridis', edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Win Rate (%)', fontweight='bold')
    ax6.set_ylabel('Bot Return (%)', fontweight='bold')
    ax6.set_title('Win Rate vs Returns (size = # trades)', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax6.grid(True, alpha=0.2)
    ax6.set_facecolor('#F8F9FA')

    # 7. Return Distribution by Stock
    ax7 = plt.subplot(4, 3, 7)
    ax7.hist(results_df['Bot Return (%)'], bins=25, color=GS_BLUE, alpha=0.7, edgecolor='black')
    ax7.axvline(x=0, color=GS_RED, linestyle='--', linewidth=2, label='Breakeven', alpha=0.8)
    ax7.axvline(x=results_df['Bot Return (%)'].mean(), color=GS_GREEN, linestyle='--',
               linewidth=2, label=f'Mean: {results_df["Bot Return (%)"].mean():.2f}%', alpha=0.8)
    ax7.axvline(x=results_df['Bot Return (%)'].median(), color=GS_GOLD, linestyle='--',
               linewidth=2, label=f'Median: {results_df["Bot Return (%)"].median():.2f}%', alpha=0.8)
    ax7.set_xlabel('Return (%)', fontweight='bold')
    ax7.set_ylabel('Frequency', fontweight='bold')
    ax7.set_title('Return Distribution Across All Stocks', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.2, axis='y')
    ax7.set_facecolor('#F8F9FA')

    # 8. Sharpe Ratio Ranking
    ax8 = plt.subplot(4, 3, 8)
    sharpe_sorted = results_df.nlargest(15, 'Sharpe Ratio')
    colors_sharpe = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sharpe_sorted)))
    ax8.barh(range(len(sharpe_sorted)), sharpe_sorted['Sharpe Ratio'],
            color=colors_sharpe, edgecolor='black', linewidth=0.5)
    ax8.set_yticks(range(len(sharpe_sorted)))
    ax8.set_yticklabels(sharpe_sorted['Ticker'], fontsize=9)
    ax8.set_xlabel('Sharpe Ratio', fontweight='bold')
    ax8.set_title('Top 15: Best Risk-Adjusted Returns', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.2, axis='x')
    ax8.set_facecolor('#F8F9FA')
    for i, v in enumerate(sharpe_sorted['Sharpe Ratio']):
        ax8.text(v, i, f'  {v:.2f}', va='center', fontsize=8, fontweight='bold')

    # 9. Trading Efficiency
    ax9 = plt.subplot(4, 3, 9)
    results_df['Return_per_Trade'] = results_df['Bot Return (%)'] / results_df['Total Trades'].replace(0, 1)
    eff_sorted = results_df.nlargest(15, 'Return_per_Trade')
    ax9.barh(range(len(eff_sorted)), eff_sorted['Return_per_Trade'],
            color=GS_LIGHT_BLUE, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax9.set_yticks(range(len(eff_sorted)))
    ax9.set_yticklabels(eff_sorted['Ticker'], fontsize=9)
    ax9.set_xlabel('Return per Trade (%)', fontweight='bold')
    ax9.set_title('Top 15: Most Efficient Trading', fontsize=11, fontweight='bold', color=GS_BLUE)
    ax9.invert_yaxis()
    ax9.grid(True, alpha=0.2, axis='x')
    ax9.set_facecolor('#F8F9FA')

    # 10. Comprehensive Performance Table
    ax10 = plt.subplot(4, 3, 10)
    ax10.axis('off')

    outperformed = len(results_df[results_df['Excess Return (%)'] > 0])
    positive_returns = len(results_df[results_df['Bot Return (%)'] > 0])

    perf_table = f"""
╔══════════════════════════════════════════════╗
║   MULTI-STOCK PERFORMANCE ANALYTICS          ║
╚══════════════════════════════════════════════╝

PORTFOLIO SUMMARY
─────────────────────────────────────────────
Total Stocks Analyzed:      {len(results_df)}
Total Capital Deployed:     ${config['initial_capital'] * len(results_df):,.0f}

RETURN METRICS
─────────────────────────────────────────────
Mean Bot Return:            {results_df['Bot Return (%)'].mean():+.2f}%
Median Bot Return:          {results_df['Bot Return (%)'].median():+.2f}%
Std Dev Returns:            {results_df['Bot Return (%)'].std():.2f}%

Best Performer:             {results_df.nlargest(1, 'Bot Return (%)').iloc[0]['Ticker']} ({results_df['Bot Return (%)'].max():+.2f}%)
Worst Performer:            {results_df.nsmallest(1, 'Bot Return (%)').iloc[0]['Ticker']} ({results_df['Bot Return (%)'].min():+.2f}%)

STRATEGY EFFECTIVENESS
─────────────────────────────────────────────
Profitable Stocks:          {positive_returns}/{len(results_df)} ({positive_returns/len(results_df)*100:.1f}%)
Beat Buy & Hold:            {outperformed}/{len(results_df)} ({outperformed/len(results_df)*100:.1f}%)
Avg Excess Return:          {results_df['Excess Return (%)'].mean():+.2f}%

RISK METRICS
─────────────────────────────────────────────
Avg Sharpe Ratio:           {results_df['Sharpe Ratio'].mean():.2f}
Avg Max Drawdown:           {results_df['Max Drawdown (%)'].mean():.2f}%
Avg Win Rate:               {results_df['Win Rate'].mean():.2f}%

TRADING ACTIVITY
─────────────────────────────────────────────
Avg Trades per Stock:       {results_df['Total Trades'].mean():.1f}
Total Trades Executed:      {results_df['Total Trades'].sum():.0f}
    """

    ax10.text(0.05, 0.5, perf_table, fontsize=8.5, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#F0F8FF',
             edgecolor=GS_BLUE, linewidth=2, alpha=0.8))

    # 11. Top Winners Detail
    ax11 = plt.subplot(4, 3, 11)
    ax11.axis('tight')
    ax11.axis('off')

    top10_detail = results_df.nlargest(10, 'Bot Return (%)')[['Ticker', 'Bot Return (%)',
                                                                 'Win Rate', 'Sharpe Ratio',
                                                                 'Total Trades']]
    top10_detail.columns = ['Stock', 'Return%', 'Win%', 'Sharpe', 'Trades']

    table = ax11.table(cellText=top10_detail.round(2).values,
                      colLabels=top10_detail.columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Style header
    for i in range(len(top10_detail.columns)):
        table[(0, i)].set_facecolor(GS_BLUE)
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, len(top10_detail) + 1):
        for j in range(len(top10_detail.columns)):
            if j == 0:  # Ticker column
                table[(i, j)].set_facecolor('#E8F4F8')
            table[(i, j)].set_edgecolor('gray')

    ax11.set_title('Top 10 Performers - Detailed Metrics', fontsize=11,
                  fontweight='bold', color=GS_BLUE, pad=20)

    # 12. Bottom Performers Detail
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('tight')
    ax12.axis('off')

    bottom10_detail = results_df.nsmallest(10, 'Bot Return (%)')[['Ticker', 'Bot Return (%)',
                                                                     'Win Rate', 'Max Drawdown (%)',
                                                                     'Total Trades']]
    bottom10_detail.columns = ['Stock', 'Return%', 'Win%', 'Drawdown%', 'Trades']

    table2 = ax12.table(cellText=bottom10_detail.round(2).values,
                       colLabels=bottom10_detail.columns,
                       cellLoc='center',
                       loc='center',
                       bbox=[0, 0, 1, 1])
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1, 1.8)

    # Style header
    for i in range(len(bottom10_detail.columns)):
        table2[(0, i)].set_facecolor(GS_RED)
        table2[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, len(bottom10_detail) + 1):
        for j in range(len(bottom10_detail.columns)):
            if j == 0:  # Ticker column
                table2[(i, j)].set_facecolor('#FFF0F0')
            table2[(i, j)].set_edgecolor('gray')

    ax12.set_title('Bottom 10 Performers - Risk Analysis', fontsize=11,
                  fontweight='bold', color=GS_RED, pad=20)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Add footer
    fig.text(0.5, 0.005, 'Goldman Sachs-Quality Trading Analysis | For Educational Purposes Only',
            ha='center', fontsize=9, style='italic', color=GS_GRAY)

    filename = 'multi_stock_trading_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Professional visualization saved as '{filename}'")
    plt.show()


def main():
    """Main execution function"""

    # Configuration
    config = {
        'initial_capital': 10000,
        'max_position_size': 0.25,
        'min_position_size': 0.10,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.10,
        'start_date': None,
        'end_date': None,
        'verbose': False
    }

    # Run simulation
    results_df = run_multi_stock_simulation(SP100_TICKERS, config, max_workers=20)

    if len(results_df) > 0:
        # Save results to CSV
        results_df.to_csv('multi_stock_results.csv', index=False)
        print(f"Results saved to 'multi_stock_results.csv'\n")

        # Generate report
        generate_summary_report(results_df)

        # Visualize
        visualize_multi_stock_results(results_df, config)

        print(f"\nSimulation complete!")
        print(f"- Results CSV: multi_stock_results.csv")
        print(f"- Visualization: multi_stock_trading_results.png")
    else:
        print("No successful simulations to report.")


if __name__ == "__main__":
    main()