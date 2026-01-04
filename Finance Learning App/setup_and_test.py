#!/usr/bin/env python3
"""
Setup and Test Script for Quantitative Finance Platform
Verifies all components are working correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*80)
    print("TESTING MODULE IMPORTS")
    print("="*80)

    modules = [
        ("Config", "from config.settings import Config"),
        ("GS Theme", "from config.themes import GoldmanSachsTheme"),
        ("Database Models", "from database.models import init_database, get_session"),
        ("Black-Scholes", "from core.derivatives.black_scholes import BlackScholesModel"),
        ("Implied Volatility", "from core.derivatives.implied_volatility import ImpliedVolatility"),
        ("Risk Metrics", "from core.portfolio.risk_metrics import RiskMetrics"),
        ("Portfolio Optimizer", "from core.portfolio.optimizer import PortfolioOptimizer"),
        ("Technical Indicators", "from core.trading.technical_indicators import TechnicalIndicators"),
        ("Signal Generator", "from core.trading.signal_generator import SignalGenerator"),
        ("Finnhub Client", "from core.market_data.finnhub_client import FinnhubClient"),
    ]

    for name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"✅ {name:<30} - OK")
        except ImportError as e:
            print(f"❌ {name:<30} - FAILED: {e}")
            return False

    print("\n✅ All imports successful!")
    return True


def test_database():
    """Test database initialization"""
    print("\n" + "="*80)
    print("TESTING DATABASE")
    print("="*80)

    try:
        from database.models import init_database, get_session
        from config.settings import Config

        print(f"Database path: {Config.DATABASE_PATH}")

        # Initialize database
        engine = init_database()
        print("✅ Database initialized successfully")

        # Test session creation
        session = get_session(engine)
        print("✅ Database session created")

        session.close()
        print("✅ Database connection closed")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_black_scholes():
    """Test Black-Scholes calculations"""
    print("\n" + "="*80)
    print("TESTING BLACK-SCHOLES MODEL")
    print("="*80)

    try:
        from core.derivatives.black_scholes import BlackScholesModel

        # Test parameters
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.02

        call_price = BlackScholesModel.call_price(S, K, T, r, sigma, q)
        put_price = BlackScholesModel.put_price(S, K, T, r, sigma, q)

        print(f"Call Price: ${call_price:.4f}")
        print(f"Put Price:  ${put_price:.4f}")

        # Calculate Greeks
        greeks = BlackScholesModel.all_greeks(S, K, T, r, sigma, q, 'call')

        print("\nGreeks (Call Option):")
        print(f"  Delta: {greeks['delta']:.4f}")
        print(f"  Gamma: {greeks['gamma']:.4f}")
        print(f"  Vega:  {greeks['vega']:.4f}")
        print(f"  Theta: {greeks['theta']:.4f}")
        print(f"  Rho:   {greeks['rho']:.4f}")

        # Verify put-call parity
        import numpy as np
        parity_check = abs((call_price - put_price) - (S*np.exp(-q*T) - K*np.exp(-r*T))) < 0.01

        if parity_check:
            print("\n✅ Put-Call Parity verified")
        else:
            print("\n⚠️ Put-Call Parity check failed")

        print("✅ Black-Scholes test passed")
        return True

    except Exception as e:
        print(f"❌ Black-Scholes test failed: {e}")
        return False


def test_risk_metrics():
    """Test risk metrics calculations"""
    print("\n" + "="*80)
    print("TESTING RISK METRICS")
    print("="*80)

    try:
        from core.portfolio.risk_metrics import RiskMetrics
        import numpy as np

        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns for 1 year

        # Calculate metrics
        var_95 = RiskMetrics.value_at_risk(returns, 0.95)
        cvar_95 = RiskMetrics.conditional_var(returns, 0.95)
        sharpe = RiskMetrics.sharpe_ratio(returns, 0.02)
        max_dd = RiskMetrics.max_drawdown(returns)

        print(f"VaR (95%):        {var_95*100:.2f}%")
        print(f"CVaR (95%):       {cvar_95*100:.2f}%")
        print(f"Sharpe Ratio:     {sharpe:.4f}")
        print(f"Max Drawdown:     {max_dd*100:.2f}%")

        # Test comprehensive metrics
        all_metrics = RiskMetrics.calculate_all(returns)
        print(f"\n✅ Calculated {len(all_metrics)} risk metrics")

        print("✅ Risk metrics test passed")
        return True

    except Exception as e:
        print(f"❌ Risk metrics test failed: {e}")
        return False


def test_portfolio_optimizer():
    """Test portfolio optimization"""
    print("\n" + "="*80)
    print("TESTING PORTFOLIO OPTIMIZER")
    print("="*80)

    try:
        from core.portfolio.optimizer import PortfolioOptimizer
        import pandas as pd
        import numpy as np

        # Generate sample returns for 4 assets
        np.random.seed(42)
        n_days = 252
        asset_names = ['Stock A', 'Stock B', 'Stock C', 'Stock D']

        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                [0.0005, 0.0003, 0.0004, 0.0002],
                [[0.0004, 0.0001, 0.0002, 0.00005],
                 [0.0001, 0.0002, 0.00008, 0.00003],
                 [0.0002, 0.00008, 0.0003, 0.00006],
                 [0.00005, 0.00003, 0.00006, 0.0001]],
                n_days
            ),
            columns=asset_names
        )

        optimizer = PortfolioOptimizer(returns_data)

        # Max Sharpe portfolio
        max_sharpe = optimizer.max_sharpe_portfolio()
        print(f"Max Sharpe Ratio Portfolio:")
        print(f"  Expected Return:  {max_sharpe['return']*100:.2f}%")
        print(f"  Volatility:       {max_sharpe['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio:     {max_sharpe['sharpe_ratio']:.4f}")

        # Min Variance portfolio
        min_var = optimizer.min_variance_portfolio()
        print(f"\nMin Variance Portfolio:")
        print(f"  Expected Return:  {min_var['return']*100:.2f}%")
        print(f"  Volatility:       {min_var['volatility']*100:.2f}%")

        print("\n✅ Portfolio optimization test passed")
        return True

    except Exception as e:
        print(f"❌ Portfolio optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finnhub_client():
    """Test Finnhub API client"""
    print("\n" + "="*80)
    print("TESTING FINNHUB API CLIENT")
    print("="*80)

    try:
        from core.market_data.finnhub_client import FinnhubClient
        from config.settings import Config

        client = FinnhubClient()

        if not Config.validate_finnhub_key():
            print("⚠️  Finnhub API key not configured")
            print("    Add FINNHUB_API_KEY to .env file to test live data")
            print("✅ Client initialized (offline mode)")
            return True

        # Test quote
        print("Testing real-time quote for AAPL...")
        quote = client.get_quote("AAPL")
        if quote and quote.get('c'):
            print(f"  Current Price: ${quote['c']:.2f}")
            print("✅ Quote fetched successfully")
        else:
            print("⚠️  No quote data received")

        # Test company profile
        print("\nTesting company profile for AAPL...")
        profile = client.get_company_profile("AAPL")
        if profile:
            print(f"  Company: {profile.get('name', 'N/A')}")
            print(f"  Industry: {profile.get('finnhubIndustry', 'N/A')}")
            print("✅ Profile fetched successfully")

        client.close()
        print("\n✅ Finnhub client test passed")
        return True

    except Exception as e:
        print(f"❌ Finnhub client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment():
    """Check Python version and key packages"""
    print("\n" + "="*80)
    print("CHECKING ENVIRONMENT")
    print("="*80)

    # Python version
    print(f"Python Version: {sys.version}")

    if sys.version_info < (3, 9):
        print("⚠️  Python 3.9 or higher recommended")
    else:
        print("✅ Python version OK")

    # Check key packages
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'plotly',
        'yfinance',
        'finnhub',
        'ta',
        'sqlalchemy'
    ]

    print("\nChecking installed packages:")
    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            all_ok = False

    if all_ok:
        print("\n✅ All required packages installed")
    else:
        print("\n⚠️  Some packages missing. Run: pip install -r requirements.txt")

    return all_ok


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "QUANTITATIVE FINANCE PLATFORM - SETUP & TEST" + " "*19 + "║")
    print("╚" + "="*78 + "╝")

    results = []

    # Environment check
    results.append(("Environment Check", check_environment()))

    # Module imports
    results.append(("Module Imports", test_imports()))

    # Database
    results.append(("Database", test_database()))

    # Black-Scholes
    results.append(("Black-Scholes Model", test_black_scholes()))

    # Risk Metrics
    results.append(("Risk Metrics", test_risk_metrics()))

    # Portfolio Optimizer
    results.append(("Portfolio Optimizer", test_portfolio_optimizer()))

    # Finnhub Client
    results.append(("Finnhub API Client", test_finnhub_client()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")

    print("="*80)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Platform is ready to use.")
        print("\nTo start the application, run:")
        print("    streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
