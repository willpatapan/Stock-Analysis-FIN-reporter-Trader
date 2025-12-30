#!/usr/bin/env python3
"""
Real-Time Stock Quote Tool using Finnhub API
Displays live stock prices without historical analysis
"""

import finnhub
import os
import sys
from datetime import datetime

# Your Finnhub API key
API_KEY = "d59laa1r01qgqlm0vr80d59laa1r01qgqlm0vr8g"

def get_realtime_quote(ticker):
    """Fetch real-time stock quote from Finnhub"""
    try:
        client = finnhub.Client(api_key=API_KEY)
        quote = client.quote(ticker)
        
        if not quote or quote.get('c') == 0:
            print(f"❌ No data available for {ticker}")
            return None
            
        return quote
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def get_company_profile(ticker):
    """Get company information"""
    try:
        client = finnhub.Client(api_key=API_KEY)
        profile = client.company_profile2(symbol=ticker)
        return profile
    except:
        return None

def display_quote(ticker):
    """Display formatted real-time quote"""
    print("\n" + "="*80)
    print(f"🔴 LIVE STOCK QUOTE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Get company profile
    profile = get_company_profile(ticker)
    if profile:
        print(f"\n📊 {profile.get('name', ticker)} ({ticker})")
        if 'exchange' in profile:
            print(f"Exchange: {profile['exchange']}")
        if 'marketCapitalization' in profile:
            print(f"Market Cap: ${profile['marketCapitalization']:.2f}B")
    else:
        print(f"\n📊 {ticker}")
    
    # Get quote
    quote = get_realtime_quote(ticker)
    if not quote:
        return
    
    current = quote['c']
    change = quote['d']
    change_pct = quote['dp']
    high = quote['h']
    low = quote['l']
    open_price = quote['o']
    prev_close = quote['pc']
    
    # Determine if market is up or down
    direction = "🟢" if change >= 0 else "🔴"
    
    print(f"\n{direction} CURRENT PRICE: ${current:.2f}")
    print(f"   Change: ${change:.2f} ({change_pct:+.2f}%)")
    print(f"\n📈 TODAY'S RANGE:")
    print(f"   Open: ${open_price:.2f}")
    print(f"   High: ${high:.2f}")
    print(f"   Low: ${low:.2f}")
    print(f"\n📅 PREVIOUS CLOSE: ${prev_close:.2f}")
    
    # Trading signals based on price action
    print(f"\n💡 QUICK ANALYSIS:")
    
    if change_pct > 2:
        print("   • Strong upward momentum (+2%+)")
    elif change_pct > 0.5:
        print("   • Positive movement")
    elif change_pct < -2:
        print("   • Significant decline (-2%+)")
    elif change_pct < -0.5:
        print("   • Negative movement")
    else:
        print("   • Trading relatively flat")
    
    range_pct = ((high - low) / current) * 100
    print(f"   • Intraday volatility: {range_pct:.2f}%")
    
    current_position = ((current - low) / (high - low)) * 100 if high != low else 50
    if current_position > 70:
        print(f"   • Currently near day's high ({current_position:.0f}% of range)")
    elif current_position < 30:
        print(f"   • Currently near day's low ({current_position:.0f}% of range)")
    else:
        print(f"   • Currently mid-range ({current_position:.0f}% of range)")
    
    print("\n" + "="*80)
    print("Data powered by Finnhub.io | Real-time quotes")
    print("="*80)

def watch_mode(tickers):
    """Watch multiple stocks continuously"""
    import time
    
    print("\n🔄 WATCH MODE - Refreshing every 10 seconds (Press Ctrl+C to stop)")
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            for ticker in tickers:
                quote = get_realtime_quote(ticker)
                if quote:
                    change_pct = quote['dp']
                    direction = "🟢" if quote['d'] >= 0 else "🔴"
                    print(f"{direction} {ticker:6s} ${quote['c']:8.2f}  {change_pct:+6.2f}%")
            
            print("\n(Refreshing in 10s... Press Ctrl+C to stop)")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n✋ Watch mode stopped")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} AAPL              # Single quote")
        print(f"  {sys.argv[0]} AAPL MSFT GOOGL   # Multiple quotes")
        print(f"  {sys.argv[0]} watch AAPL MSFT   # Watch mode (auto-refresh)")
        sys.exit(1)
    
    # Check for watch mode
    if sys.argv[1].lower() == 'watch':
        if len(sys.argv) < 3:
            print("❌ Please specify tickers to watch")
            sys.exit(1)
        watch_mode([t.upper() for t in sys.argv[2:]])
        return
    
    # Display quotes for each ticker
    tickers = [t.upper() for t in sys.argv[1:]]
    
    for ticker in tickers:
        display_quote(ticker)
        if len(tickers) > 1:
            print("\n")

if __name__ == "__main__":
    main()
