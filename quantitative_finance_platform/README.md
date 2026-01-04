# Quantitative Finance Platform

A professional-grade finance education and trading platform featuring interactive learning modules, paper trading simulation, institutional-quality quantitative tools, and a Goldman Sachs-inspired professional interface.

## Features

### Learning & Education
- **Interactive Learning Modules**: Comprehensive curriculum from fundamentals to advanced quantitative finance
- **Assessment System**: Progress tracking with instant feedback and performance analytics
- **Progressive Curriculum**: Structured learning path with prerequisite-based module unlocking

### Quantitative Tools
- **Black-Scholes Options Pricing**: European call and put pricing with complete Greeks analysis
- **Portfolio Optimization**: Modern Portfolio Theory (MPT) implementation with efficient frontier generation
- **Risk Management**: Value at Risk (VaR), Conditional VaR (CVaR), Sharpe Ratio, and Maximum Drawdown calculations
- **Technical Analysis**: Suite of 15+ technical indicators including RSI, MACD, Bollinger Bands, and Ichimoku Cloud

### Investment Banking
- **M&A Analysis**: Accretion and dilution modeling for merger transactions
- **Valuation Models**: Discounted Cash Flow (DCF) and Comparable Company Analysis frameworks
- **Presentation Tools**: Professional pitch deck generation capabilities

### Trading Simulation
- **Paper Trading**: Virtual trading environment with $100,000 starting capital
- **Real-Time Market Data**: Live data integration via Finnhub API
- **Strategy Implementation**: Apply learned strategies in simulated environment
- **Performance Analytics**: Institutional-grade performance measurement and attribution

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Finnhub API key (free tier available at https://finnhub.io/register)

### Installation

1. Navigate to the project directory:
   ```bash
   cd quantitative_finance_platform
   ```

2. Create and activate virtual environment (recommended):
   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your Finnhub API key
   # Obtain a free key at: https://finnhub.io/register
   ```

5. Initialize the database:
   ```bash
   python database/models.py
   ```

6. Launch the application:
   ```bash
   streamlit run app.py
   ```

7. Access the platform at `http://localhost:8501`

## Project Structure

```
quantitative_finance_platform/
├── app.py                          # Main application entry point
├── config/
│   ├── settings.py                 # Application configuration
│   ├── themes.py                   # Goldman Sachs theme definitions
│   └── learning_modules.json       # Educational content repository
├── core/                           # Core quantitative engines
│   ├── derivatives/
│   │   ├── black_scholes.py       # Options pricing models
│   │   └── implied_volatility.py   # Implied volatility calculator
│   ├── portfolio/
│   │   ├── optimizer.py           # Portfolio optimization algorithms
│   │   └── risk_metrics.py        # Risk analysis framework
│   ├── trading/
│   │   ├── technical_indicators.py # Technical analysis library
│   │   └── signal_generator.py    # Trading signal generation
│   └── market_data/
│       └── finnhub_client.py      # Market data API client
├── pages/                          # Streamlit page modules
│   ├── 1_Home.py
│   ├── 2_Learning_Hub.py
│   ├── 3_Classic_Strategies.py
│   ├── 4_Advanced_Quant.py
│   ├── 5_Investment_Banking.py
│   ├── 6_Paper_Trading.py
│   ├── 7_Portfolio_Analysis.py
│   ├── 8_IPO_Tracker.py
│   └── 9_Deck_Generator.py
├── components/                     # Reusable UI components
├── database/
│   └── models.py                  # SQLite database schema
└── utils/
    └── formatters.py              # Utility functions
```

## API Configuration

### Finnhub API (Required for Live Data)
1. Register at [finnhub.io](https://finnhub.io/register)
2. Obtain your API key from the dashboard
3. Add to `.env` file:
   ```
   FINNHUB_API_KEY=your_key_here
   ```

**Free Tier Capabilities:**
- 60 API calls per minute
- Real-time stock quotes
- Company profiles and fundamentals
- Market news feed
- IPO calendar access

## Curriculum Structure

### Beginner Level (Modules 1-4)
1. Introduction to Stock Markets
2. Understanding Risk and Return
3. Technical Analysis Fundamentals
4. Fundamental Analysis Basics

### Intermediate Level (Modules 5-7)
5. Portfolio Diversification Strategies
6. Options Pricing with Black-Scholes
7. Modern Portfolio Theory

### Advanced Level (Modules 8-10)
8. Value Investing Strategies
9. Momentum Trading Systems
10. Risk Management Techniques

## Available Tools

### Options Calculator
- Black-Scholes pricing model implementation
- Greeks analysis (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility calculation
- Profit and loss diagram generation

### Portfolio Optimizer
- Maximum Sharpe Ratio optimization
- Minimum Variance portfolio construction
- Risk Parity allocation
- Efficient Frontier visualization

### Risk Analytics
- Value at Risk (VaR) - Historical, Parametric, and Monte Carlo methods
- Conditional Value at Risk (CVaR)
- Sharpe and Sortino Ratio calculation
- Maximum Drawdown analysis
- Beta and Alpha measurement

### Trading Strategies
- Value Investing framework
- Growth Investing methodology
- Momentum Trading systems
- Mean Reversion strategies
- Pairs Trading analysis

## Design Philosophy

### Visual Identity
- **Primary Color**: Goldman Sachs Blue (#0033A0)
- **Accent Color**: Goldman Sachs Gold (#C9A961)
- **Layout**: Clean, institutional design with emphasis on data clarity
- **Visualizations**: Publication-quality charts and graphics

### Development Principles
- Modular architecture for maintainability
- Type safety and error handling
- Comprehensive documentation
- Professional-grade code quality

## Development Status

### Phase 1: Foundation (Complete)
- Project architecture and structure
- Database schema implementation
- API integration layer
- Core quantitative modules
- Goldman Sachs theme implementation
- Main application framework

### Phase 2: Learning Hub & Classic Strategies (Complete)
- Learning module content (3 modules)
- Quiz engine with scoring system
- Progress tracking database
- Classic strategy backtesting (4 strategies)
- Performance metrics visualization

### Phase 3: Advanced Tools (Planned)
- Paper trading simulator
- Advanced quantitative tools
- Real-time portfolio tracking

### Phase 4: Investment Banking (Planned)
- M&A analyzer interface
- DCF valuation tools
- Presentation deck generator

## Contributing

This is an educational project designed to demonstrate quantitative finance concepts and software engineering best practices. Contributions that enhance educational value or code quality are welcome.

## Disclaimer

**FOR EDUCATIONAL PURPOSES ONLY**

This platform is designed exclusively for learning and simulation purposes. It is NOT:
- Financial advice or investment recommendations
- A recommendation to buy or sell any securities
- Suitable for actual trading without comprehensive risk assessment and proper licensing

All strategies and models are provided for educational demonstration only. Historical performance does not guarantee future results. Users should conduct independent research and consult with licensed financial advisors before making any investment decisions.

The platform creators assume no liability for financial losses incurred through misuse of this educational tool.

## License

Educational use only. Not licensed for commercial distribution or production trading applications.

## Acknowledgments

- **Framework**: Built with [Streamlit](https://streamlit.io/)
- **Market Data**: Provided by [Finnhub](https://finnhub.io/)
- **Technical Analysis**: Powered by [ta library](https://github.com/bukosabino/ta)
- **Design Inspiration**: Institutional finance industry standards

## Technical Support

For technical issues or questions:
1. Review the inline documentation and code comments
2. Consult the LAUNCH_GUIDE.md for detailed usage instructions
3. Verify all dependencies are correctly installed
4. Test with sample data before using custom inputs

---

**Version**: 1.0.0
**Last Updated**: January 2026
**Author**: Will Patapan
**Status**: Phase 2 Complete - Production Ready for Educational Use
