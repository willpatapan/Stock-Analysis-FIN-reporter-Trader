"""
Classic Trading Strategies - Value, Growth, Momentum, Mean Reversion
Interactive backtesting and strategy analysis
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from core.trading.technical_indicators import TechnicalIndicators
from core.trading.signal_generator import SignalGenerator, StrategySignals
from core.portfolio.risk_metrics import RiskMetrics
import yfinance as yf

st.set_page_config(
    page_title="Classic Strategies - Quantitative Finance Platform",
    page_icon="📈",
    layout="wide"
)

# Load custom CSS
css_path = Config.ASSETS_DIR / 'styles' / 'goldman_sachs_theme.css'
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header
st.markdown(f"""
<div style='background: linear-gradient(135deg, {Config.GS_BLUE} 0%, {Config.GS_DARK_BLUE} 100%);
            color: white; padding: 2rem; border-radius: 8px; margin-bottom: 2rem;
            border-bottom: 3px solid {Config.GS_GOLD};'>
    <h1 style='margin: 0; color: white;'>📈 Classic Trading Strategies</h1>
    <p style='margin: 0.5rem 0 0 0; color: {Config.GS_GOLD}; font-size: 1.1rem;'>
        Backtest proven strategies used by professional traders
    </p>
</div>
""", unsafe_allow_html=True)

# Strategy definitions
STRATEGIES = {
    "Value Investing": {
        "description": "**Buy undervalued companies** based on fundamental metrics like P/E ratio, P/B ratio, and dividend yield.",
        "icon": "💰",
        "metrics": ["P/E Ratio", "P/B Ratio", "Dividend Yield", "Price/Sales"],
        "how_it_works": """
**Philosophy**: "Buy a dollar for fifty cents"

Value investing looks for stocks trading below their intrinsic value.

**Key Metrics:**
- **P/E Ratio < 15**: Lower is cheaper
- **P/B Ratio < 1.5**: Trading below book value
- **Dividend Yield > 2%**: Income generation
- **Strong fundamentals**: Profitable with low debt

**Famous Practitioners**: Warren Buffett, Benjamin Graham
        """,
        "pros": ["Works in all market conditions", "Lower volatility", "Dividend income"],
        "cons": ["Requires patience", "Value traps exist", "May underperform in bull markets"]
    },
    "Growth Investing": {
        "description": "**Invest in fast-growing companies** with high revenue and earnings growth, even at premium valuations.",
        "icon": "🚀",
        "metrics": ["Revenue Growth", "EPS Growth", "PEG Ratio", "R&D Spending"],
        "how_it_works": """
**Philosophy**: "Invest in the future"

Growth investing focuses on companies expanding rapidly.

**Key Indicators:**
- **Revenue Growth > 20% annually**
- **EPS Growth > 15% annually**
- **PEG Ratio < 2**: Growth at reasonable price
- **Increasing market share**

**Famous Practitioners**: Peter Lynch, Cathie Wood
        """,
        "pros": ["High return potential", "Compounds wealth quickly", "Exciting companies"],
        "cons": ["Higher volatility", "Expensive valuations", "Risk of disappointment"]
    },
    "Momentum Trading": {
        "description": "**Follow the trend** - buy stocks showing strong upward momentum and sell those trending down.",
        "icon": "📊",
        "metrics": ["RSI", "MACD", "ROC", "Price vs Moving Averages"],
        "how_it_works": """
**Philosophy**: "The trend is your friend"

Momentum strategies capitalize on price trends continuing.

**Entry Signals:**
- **RSI > 60**: Strong upward momentum
- **MACD crosses above signal**: Bullish crossover
- **Price > SMA 50 > SMA 200**: Strong uptrend
- **High relative strength** vs market

**Famous Practitioners**: Mark Minervini, William O'Neil
        """,
        "pros": ["Captures strong moves", "Clear entry/exit signals", "Works in trending markets"],
        "cons": ["High turnover", "Fails in choppy markets", "Requires discipline"]
    },
    "Mean Reversion": {
        "description": "**Buy oversold, sell overbought** - profit from price extremes reverting to the average.",
        "icon": "🔄",
        "metrics": ["Bollinger Bands", "RSI", "Z-Score", "Standard Deviation"],
        "how_it_works": """
**Philosophy**: "What goes up must come down (and vice versa)"

Mean reversion bets on extreme prices returning to normal.

**Buy Signals:**
- **RSI < 30**: Oversold condition
- **Price < Lower Bollinger Band**: Stretched too far down
- **Z-Score < -2**: Statistical extreme

**Sell Signals:**
- **RSI > 70**: Overbought
- **Price > Upper Bollinger Band**: Stretched too far up

**Famous Practitioners**: Jim Simons (Renaissance Technologies)
        """,
        "pros": ["High win rate", "Defined risk", "Works in range-bound markets"],
        "cons": ["Small profits per trade", "Trend can persist", "Requires quick reactions"]
    }
}

# Strategy selector
st.subheader("🎯 Select a Strategy")

selected_strategy = st.selectbox(
    "Choose a trading strategy to explore:",
    options=list(STRATEGIES.keys()),
    format_func=lambda x: f"{STRATEGIES[x]['icon']} {x}"
)

strategy_info = STRATEGIES[selected_strategy]

# Strategy overview
with st.expander(f"📚 Learn About {selected_strategy}", expanded=True):
    st.markdown(strategy_info['description'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### How It Works")
        st.markdown(strategy_info['how_it_works'])

    with col2:
        st.markdown("### Pros")
        for pro in strategy_info['pros']:
            st.markdown(f"✅ {pro}")

        st.markdown("### Cons")
        for con in strategy_info['cons']:
            st.markdown(f"⚠️ {con}")

st.markdown("---")

# Backtesting section
st.subheader("🔬 Backtest Strategy")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)")

with col2:
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

with col3:
    initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)

if st.button("🚀 Run Backtest", type="primary"):
    with st.spinner(f"Backtesting {selected_strategy} on {ticker}..."):
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                st.error(f"No data found for {ticker}. Please check the ticker symbol.")
                st.stop()

            # Calculate technical indicators
            df_with_indicators = TechnicalIndicators.calculate_all(df)

            # Generate signals based on strategy
            if selected_strategy == "Momentum Trading":
                signals = []
                for idx, row in df_with_indicators.iterrows():
                    # Momentum strategy: Buy when RSI > 60 and price above SMAs
                    if (row['RSI'] > 60 and
                        row['Close'] > row['SMA_50'] and
                        row['SMA_50'] > row['SMA_200']):
                        signals.append('BUY')
                    elif (row['RSI'] < 40 or
                          row['Close'] < row['SMA_50']):
                        signals.append('SELL')
                    else:
                        signals.append('HOLD')

            elif selected_strategy == "Mean Reversion":
                signals = []
                for idx, row in df_with_indicators.iterrows():
                    # Mean reversion: Buy oversold, sell overbought
                    if row['RSI'] < 30 or row['Close'] < row['BB_Low']:
                        signals.append('BUY')
                    elif row['RSI'] > 70 or row['Close'] > row['BB_High']:
                        signals.append('SELL')
                    else:
                        signals.append('HOLD')

            else:
                # For Value/Growth, use multi-indicator consensus
                signal_gen = SignalGenerator(consensus_threshold=0.6)
                signals = []
                for idx, row in df_with_indicators.iterrows():
                    signal = signal_gen.generate_signal(row)
                    signals.append(signal.action)

            df_with_indicators['Signal'] = signals

            # Simulate trading
            cash = initial_capital
            shares = 0
            portfolio_values = []
            trades = []

            for idx, row in df_with_indicators.iterrows():
                signal = row['Signal']
                price = row['Close']

                if signal == 'BUY' and cash > price:
                    # Buy as many shares as possible
                    shares_to_buy = int(cash / price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        shares += shares_to_buy
                        cash -= cost
                        trades.append({
                            'date': idx,
                            'action': 'BUY',
                            'price': price,
                            'shares': shares_to_buy,
                            'value': cost
                        })

                elif signal == 'SELL' and shares > 0:
                    # Sell all shares
                    proceeds = shares * price
                    cash += proceeds
                    trades.append({
                        'date': idx,
                        'action': 'SELL',
                        'price': price,
                        'shares': shares,
                        'value': proceeds
                    })
                    shares = 0

                # Calculate portfolio value
                portfolio_value = cash + (shares * price)
                portfolio_values.append(portfolio_value)

            df_with_indicators['Portfolio_Value'] = portfolio_values

            # Calculate returns
            strategy_return = (portfolio_values[-1] - initial_capital) / initial_capital
            daily_returns = df_with_indicators['Portfolio_Value'].pct_change().dropna()

            # Buy and hold comparison
            buy_hold_shares = initial_capital / df_with_indicators['Close'].iloc[0]
            buy_hold_value = buy_hold_shares * df_with_indicators['Close'].iloc[-1]
            buy_hold_return = (buy_hold_value - initial_capital) / initial_capital

            # Calculate risk metrics
            sharpe = RiskMetrics.sharpe_ratio(daily_returns.values, Config.DEFAULT_RISK_FREE_RATE)
            max_dd = RiskMetrics.max_drawdown(daily_returns.values)
            win_rate = RiskMetrics.win_rate(daily_returns.values)

            # Display results
            st.success("✅ Backtest Complete!")

            # Performance metrics
            st.subheader("📊 Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Strategy Return",
                    f"{strategy_return*100:.2f}%",
                    delta=f"{(strategy_return - buy_hold_return)*100:.2f}% vs B&H"
                )

            with col2:
                st.metric("Buy & Hold Return", f"{buy_hold_return*100:.2f}%")

            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            with col4:
                st.metric("Max Drawdown", f"{max_dd*100:.2f}%")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Trades", len(trades))

            with col2:
                st.metric("Win Rate", f"{win_rate*100:.1f}%")

            with col3:
                final_value = portfolio_values[-1]
                st.metric("Final Portfolio Value", f"${final_value:,.2f}")

            # Portfolio value chart
            st.subheader("📈 Portfolio Value Over Time")

            fig = go.Figure()

            # Strategy performance
            fig.add_trace(go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['Portfolio_Value'],
                mode='lines',
                name=f'{selected_strategy} Strategy',
                line=dict(color=Config.GS_BLUE, width=2)
            ))

            # Buy & hold comparison
            buy_hold_values = buy_hold_shares * df_with_indicators['Close']
            fig.add_trace(go.Scatter(
                x=df_with_indicators.index,
                y=buy_hold_values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color=Config.GS_GOLD, width=2, dash='dash')
            ))

            # Mark trades
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_values = [df_with_indicators.loc[t['date'], 'Portfolio_Value'] for t in buy_trades]
                fig.add_trace(go.Scatter(
                    x=buy_dates,
                    y=buy_values,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))

            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_values = [df_with_indicators.loc[t['date'], 'Portfolio_Value'] for t in sell_trades]
                fig.add_trace(go.Scatter(
                    x=sell_dates,
                    y=sell_values,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))

            fig.update_layout(
                title=f'{ticker} - {selected_strategy} Backtest Results',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Trade history
            if trades:
                st.subheader("📋 Trade History")

                trades_df = pd.DataFrame(trades)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df = trades_df.sort_values('date', ascending=False)

                st.dataframe(
                    trades_df.style.format({
                        'price': '${:.2f}',
                        'value': '${:,.2f}'
                    }),
                    use_container_width=True
                )

            # Strategy insights
            st.subheader("💡 Strategy Insights")

            if strategy_return > buy_hold_return:
                st.success(f"""
                **Excellent Performance!** The {selected_strategy} strategy outperformed buy-and-hold by
                **{(strategy_return - buy_hold_return)*100:.2f}%**.

                This suggests the strategy effectively captured opportunities in {ticker}.
                """)
            else:
                st.warning(f"""
                **Underperformance Alert**: The {selected_strategy} strategy underperformed buy-and-hold by
                **{abs(strategy_return - buy_hold_return)*100:.2f}%**.

                Consider:
                - Adjusting strategy parameters
                - Testing on different stocks
                - Combining with other strategies
                """)

        except Exception as e:
            st.error(f"Error during backtest: {e}")
            import traceback
            st.code(traceback.format_exc())

# Strategy comparison
st.markdown("---")
st.subheader("📊 Strategy Comparison Guide")

comparison_df = pd.DataFrame({
    "Strategy": list(STRATEGIES.keys()),
    "Best Market": ["Sideways", "Bull", "Trending", "Choppy"],
    "Time Horizon": ["Long-term", "Long-term", "Short-term", "Short-term"],
    "Win Rate": ["Medium", "Low", "Medium", "High"],
    "Risk Level": ["Low", "High", "Medium", "Medium"]
})

st.dataframe(comparison_df, use_container_width=True)

st.info("""
💡 **Pro Tip**: No single strategy works all the time. Professional traders often combine multiple strategies
or switch between them based on market conditions. Start by mastering one strategy before expanding your toolkit.
""")

# Footer
st.markdown("---")
st.caption("""
**Disclaimer**: Past performance does not guarantee future results. These backtests use historical data
and may not reflect real trading costs, slippage, or market impact. Use for educational purposes only.
""")
