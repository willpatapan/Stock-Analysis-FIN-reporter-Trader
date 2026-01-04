"""
Application Configuration Settings
Manages API keys, cache settings, and global constants
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""

    # App Metadata
    APP_NAME = "Quantitative Finance Platform"
    VERSION = "1.0.0"
    DESCRIPTION = "Professional quantitative trading and finance education platform"

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    ASSETS_DIR = BASE_DIR / "assets"
    DATABASE_PATH = DATA_DIR / "app.db"

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)

    # API Configuration
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    FINNHUB_RATE_LIMIT = 60  # calls per minute (free tier)

    # Cache Settings (in seconds)
    QUOTE_CACHE_TTL = 300  # 5 minutes
    PROFILE_CACHE_TTL = 86400  # 24 hours
    NEWS_CACHE_TTL = 900  # 15 minutes
    HISTORICAL_CACHE_TTL = 3600  # 1 hour

    # Paper Trading Configuration
    INITIAL_PAPER_CAPITAL = 100000.0  # $100,000
    MIN_TRADE_AMOUNT = 1.0  # Minimum $1 per trade

    # Learning Module Settings
    QUIZ_PASS_THRESHOLD = 0.8  # 80% to pass and unlock next module

    # Portfolio Optimization Settings
    MAX_STOCKS_IN_PORTFOLIO = 20  # Limit for performance
    DEFAULT_RISK_FREE_RATE = 0.04  # 4% annual risk-free rate

    # Backtesting Settings
    DEFAULT_INITIAL_CAPITAL = 10000.0
    DEFAULT_COMMISSION = 0.001  # 0.1% per trade

    # UI Configuration
    PAGE_TITLE = "Quantitative Finance Platform"
    PAGE_ICON = "💼"
    LAYOUT = "wide"

    # Goldman Sachs Theme Colors
    GS_BLUE = "#0033A0"
    GS_DARK_BLUE = "#002670"
    GS_GOLD = "#C9A961"
    GS_LIGHT_GRAY = "#F5F7FA"
    GS_DARK_GRAY = "#2C3E50"
    GS_SUCCESS = "#2ECC71"
    GS_DANGER = "#E74C3C"

    # Session State Keys
    SESSION_ID = "session_id"
    PORTFOLIO_ID = "portfolio_id"
    USER_PROGRESS = "user_progress"

    @classmethod
    def get_database_url(cls):
        """Get SQLAlchemy database URL"""
        return f"sqlite:///{cls.DATABASE_PATH}"

    @classmethod
    def validate_finnhub_key(cls):
        """Check if Finnhub API key is configured"""
        return bool(cls.FINNHUB_API_KEY and cls.FINNHUB_API_KEY != '')
