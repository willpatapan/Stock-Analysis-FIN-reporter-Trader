"""
Finnhub API Client with Intelligent Caching
Manages API calls with rate limiting and SQLite-backed caching
"""

import finnhub
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from config.settings import Config
from database.models import get_session, cache_market_data, get_cached_data, clear_expired_cache

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]

        if len(self.calls) >= self.calls_per_minute:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call) + 0.1  # Add 0.1s buffer
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        self.calls.append(time.time())


class FinnhubClient:
    """
    Finnhub API wrapper with caching and rate limiting

    Features:
    - Real-time stock quotes
    - Company profiles
    - Company news
    - IPO calendar
    - SQLite-backed caching with configurable TTL
    - Automatic rate limiting (60 calls/min for free tier)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub client

        Args:
            api_key: Finnhub API key (defaults to Config.FINNHUB_API_KEY)
        """
        self.api_key = api_key or Config.FINNHUB_API_KEY

        if not self.api_key:
            logger.warning("Finnhub API key not configured. Some features will be unavailable.")
            self.client = None
        else:
            self.client = finnhub.Client(api_key=self.api_key)

        self.rate_limiter = RateLimiter(calls_per_minute=Config.FINNHUB_RATE_LIMIT)
        self.session = get_session()

        # Clean up expired cache on initialization
        clear_expired_cache(self.session)

    def _make_api_call(self, func, *args, **kwargs) -> Any:
        """Wrapper for API calls with rate limiting and error handling"""
        if not self.client:
            raise ValueError("Finnhub API key not configured")

        self.rate_limiter.wait_if_needed()

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            raise

    def get_quote(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data (default: True)

        Returns:
            Dict with keys: c (current price), h (high), l (low), o (open),
                           pc (previous close), t (timestamp)
        """
        symbol = symbol.upper()

        # Check cache first
        if use_cache:
            cached = get_cached_data(self.session, symbol, 'quote')
            if cached:
                logger.debug(f"Using cached quote for {symbol}")
                return cached

        # Fetch from API
        logger.info(f"Fetching real-time quote for {symbol} from Finnhub")
        quote = self._make_api_call(self.client.quote, symbol)

        # Cache the result
        if quote and quote.get('c'):  # Verify valid data
            cache_market_data(self.session, symbol, 'quote', quote, Config.QUOTE_CACHE_TTL)

        return quote

    def get_company_profile(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get company profile information

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data (default: True)

        Returns:
            Dict with company info: name, industry, sector, marketCap, etc.
        """
        symbol = symbol.upper()

        if use_cache:
            cached = get_cached_data(self.session, symbol, 'profile')
            if cached:
                logger.debug(f"Using cached profile for {symbol}")
                return cached

        logger.info(f"Fetching company profile for {symbol} from Finnhub")
        profile = self._make_api_call(self.client.company_profile2, symbol=symbol)

        if profile:
            cache_market_data(self.session, symbol, 'profile', profile, Config.PROFILE_CACHE_TTL)

        return profile

    def get_company_news(self, symbol: str, days: int = 7, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get recent company news

        Args:
            symbol: Stock ticker symbol
            days: Number of days of news to fetch (default: 7)
            use_cache: Whether to use cached data (default: True)

        Returns:
            List of news articles with headline, summary, source, url, datetime
        """
        symbol = symbol.upper()
        cache_key = f"news_{days}d"

        if use_cache:
            cached = get_cached_data(self.session, symbol, cache_key)
            if cached:
                logger.debug(f"Using cached news for {symbol}")
                return cached

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching {days}-day news for {symbol} from Finnhub")
        news = self._make_api_call(
            self.client.company_news,
            symbol,
            _from=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d')
        )

        if news:
            cache_market_data(self.session, symbol, cache_key, news, Config.NEWS_CACHE_TTL)

        return news or []

    def get_ipo_calendar(self, from_date: Optional[str] = None, to_date: Optional[str] = None,
                        use_cache: bool = True) -> Dict[str, List[Dict]]:
        """
        Get IPO calendar

        Args:
            from_date: Start date (YYYY-MM-DD), defaults to today
            to_date: End date (YYYY-MM-DD), defaults to 3 months from now
            use_cache: Whether to use cached data (default: True)

        Returns:
            Dict with 'ipoCalendar' key containing list of IPO events
        """
        if from_date is None:
            from_date = datetime.now().strftime('%Y-%m-%d')
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')

        cache_key = f"ipo_{from_date}_{to_date}"

        if use_cache:
            cached = get_cached_data(self.session, 'IPO', cache_key)
            if cached:
                logger.debug("Using cached IPO calendar")
                return cached

        logger.info(f"Fetching IPO calendar from {from_date} to {to_date}")
        ipo_data = self._make_api_call(self.client.ipo_calendar, _from=from_date, to=to_date)

        if ipo_data:
            cache_market_data(self.session, 'IPO', cache_key, ipo_data, 3600)  # Cache for 1 hour

        return ipo_data or {'ipoCalendar': []}

    def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get analyst recommendation trends

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of recommendation trends by period
        """
        symbol = symbol.upper()
        logger.info(f"Fetching recommendation trends for {symbol}")
        return self._make_api_call(self.client.recommendation_trends, symbol) or []

    def get_price_target(self, symbol: str) -> Dict[str, Any]:
        """
        Get analyst price targets

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with targetHigh, targetLow, targetMean, targetMedian
        """
        symbol = symbol.upper()
        logger.info(f"Fetching price target for {symbol}")
        return self._make_api_call(self.client.price_target, symbol) or {}

    def search_symbol(self, query: str) -> List[Dict[str, str]]:
        """
        Search for stock symbols

        Args:
            query: Search query (company name or ticker)

        Returns:
            List of matching symbols with description, displaySymbol, symbol, type
        """
        logger.info(f"Searching for symbols matching '{query}'")
        result = self._make_api_call(self.client.symbol_lookup, query)
        return result.get('result', []) if result else []

    def get_basic_financials(self, symbol: str, metric: str = 'all') -> Dict[str, Any]:
        """
        Get company basic financials

        Args:
            symbol: Stock ticker symbol
            metric: Metric type (default: 'all')

        Returns:
            Dict with financial metrics
        """
        symbol = symbol.upper()
        logger.info(f"Fetching basic financials for {symbol}")
        return self._make_api_call(self.client.company_basic_financials, symbol, metric) or {}

    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()


# Convenience function for easy import
def get_finnhub_client() -> FinnhubClient:
    """Get a configured Finnhub client instance"""
    return FinnhubClient()


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = get_finnhub_client()

    if client.client:
        # Test quote
        print("\n=== Testing Real-Time Quote ===")
        quote = client.get_quote("AAPL")
        print(f"AAPL Quote: ${quote.get('c', 'N/A')}")

        # Test company profile
        print("\n=== Testing Company Profile ===")
        profile = client.get_company_profile("AAPL")
        print(f"Company: {profile.get('name', 'N/A')}")
        print(f"Industry: {profile.get('finnhubIndustry', 'N/A')}")

        # Test news
        print("\n=== Testing Company News ===")
        news = client.get_company_news("AAPL", days=3)
        print(f"Found {len(news)} news articles")
        if news:
            print(f"Latest: {news[0].get('headline', 'N/A')}")

        client.close()
    else:
        print("Finnhub API key not configured. Please set FINNHUB_API_KEY in .env file")
