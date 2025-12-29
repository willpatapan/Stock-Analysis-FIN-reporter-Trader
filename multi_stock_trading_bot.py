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


# S&P 100 stocks (top 100 most liquid US stocks)
SP100_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'COST', 'KO', 'AVGO', 'MCD', 'PFE', 'TMO', 'CSCO', 'ACN', 'LLY', 'DHR',
    'ADBE', 'NKE', 'ABT', 'DIS', 'WMT', 'CRM', 'VZ', 'NEE', 'TXN', 'CMCSA',
    'BMY', 'PM', 'RTX', 'ORCL', 'UPS', 'NFLX', 'INTC', 'AMD', 'QCOM', 'HON',
    'T', 'LOW', 'AMGN', 'SPGI', 'UNP', 'CAT', 'INTU', 'IBM', 'GE', 'BA',
    'SBUX', 'AMAT', 'BLK', 'MDLZ', 'CVS', 'GILD', 'AXP', 'DE', 'SCHW', 'ADI',
    'PLD', 'TJX', 'ADP', 'BKNG', 'MMC', 'REGN', 'LMT', 'CB', 'SYK', 'CI',
    'MO', 'C', 'BDX', 'ISRG', 'ZTS', 'SO', 'DUK', 'TGT', 'VRTX', 'LRCX',
    'PNC', 'USB', 'NOC', 'MMM', 'GD', 'CL', 'EOG', 'ITW', 'APD', 'SHW'
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
    """Create visualizations for multi-stock results"""

    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Bot Returns Distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(results_df['Bot Return (%)'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=results_df['Bot Return (%)'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {results_df["Bot Return (%)"].mean():.2f}%')
    ax1.set_title('Distribution of Bot Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Bot vs Buy & Hold
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(results_df['Buy & Hold Return (%)'], results_df['Bot Return (%)'], alpha=0.6, s=50)

    # Add diagonal line (y=x)
    min_val = min(results_df['Buy & Hold Return (%)'].min(), results_df['Bot Return (%)'].min())
    max_val = max(results_df['Buy & Hold Return (%)'].max(), results_df['Bot Return (%)'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance')

    ax2.set_title('Bot Return vs Buy & Hold', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Buy & Hold Return (%)')
    ax2.set_ylabel('Bot Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Excess Returns Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(results_df['Excess Return (%)'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax3.axvline(x=results_df['Excess Return (%)'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {results_df["Excess Return (%)"].mean():.2f}%')
    ax3.set_title('Distribution of Excess Returns (Bot vs Buy & Hold)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Excess Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Top 20 Performers
    ax4 = plt.subplot(3, 3, 4)
    top20 = results_df.nlargest(20, 'Bot Return (%)')
    colors = ['green' if x > 0 else 'red' for x in top20['Bot Return (%)']]
    ax4.barh(range(len(top20)), top20['Bot Return (%)'], color=colors, alpha=0.7)
    ax4.set_yticks(range(len(top20)))
    ax4.set_yticklabels(top20['Ticker'])
    ax4.set_title('Top 20 Performers (Bot Return)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Return (%)')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    # 5. Win Rate Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(results_df['Win Rate'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax5.axvline(x=results_df['Win Rate'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["Win Rate"].mean():.2f}%')
    ax5.set_title('Distribution of Win Rates', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Win Rate (%)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Sharpe Ratio Distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(results_df['Sharpe Ratio'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax6.axvline(x=results_df['Sharpe Ratio'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["Sharpe Ratio"].mean():.2f}')
    ax6.set_title('Distribution of Sharpe Ratios', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Sharpe Ratio')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Trading Activity
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(results_df['Total Trades'], results_df['Bot Return (%)'], alpha=0.6, s=50)
    ax7.set_title('Trading Activity vs Returns', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Total Trades')
    ax7.set_ylabel('Bot Return (%)')
    ax7.grid(True, alpha=0.3)

    # 8. Max Drawdown vs Returns
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(results_df['Max Drawdown (%)'], results_df['Bot Return (%)'], alpha=0.6, s=50, c=results_df['Sharpe Ratio'], cmap='viridis')
    ax8.set_title('Max Drawdown vs Returns (colored by Sharpe)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Max Drawdown (%)')
    ax8.set_ylabel('Bot Return (%)')
    plt.colorbar(ax8.collections[0], ax=ax8, label='Sharpe Ratio')
    ax8.grid(True, alpha=0.3)

    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    outperformed = len(results_df[results_df['Excess Return (%)'] > 0])
    positive_returns = len(results_df[results_df['Bot Return (%)'] > 0])

    summary_text = f"""
    MULTI-STOCK SIMULATION SUMMARY
    ══════════════════════════════════════

    Stocks Analyzed:           {len(results_df)}
    Initial Capital/Stock:     ${config['initial_capital']:,.0f}

    RETURNS:
    Avg Bot Return:            {results_df['Bot Return (%)'].mean():+.2f}%
    Avg Buy & Hold:            {results_df['Buy & Hold Return (%)'].mean():+.2f}%
    Avg Excess Return:         {results_df['Excess Return (%)'].mean():+.2f}%

    Best Bot Return:           {results_df['Bot Return (%)'].max():+.2f}%
    Worst Bot Return:          {results_df['Bot Return (%)'].min():+.2f}%

    PERFORMANCE:
    Stocks with Profit:        {positive_returns} ({positive_returns/len(results_df)*100:.1f}%)
    Bot Beat Buy & Hold:       {outperformed} ({outperformed/len(results_df)*100:.1f}%)

    Avg Win Rate:              {results_df['Win Rate'].mean():.2f}%
    Avg Sharpe Ratio:          {results_df['Sharpe Ratio'].mean():.2f}
    Avg Max Drawdown:          {results_df['Max Drawdown (%)'].mean():.2f}%
    Avg Trades/Stock:          {results_df['Total Trades'].mean():.1f}
    """

    ax9.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    filename = 'multi_stock_trading_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
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