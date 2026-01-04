"""
Database Models for Quantitative Finance Platform
SQLite schema for user progress, paper trading, and market data caching
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from config.settings import Config

Base = declarative_base()


class UserProgress(Base):
    """Track user progress through learning modules"""
    __tablename__ = 'user_progress'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    module_id = Column(String(100), nullable=False, index=True)
    completed = Column(Boolean, default=False)
    quiz_score = Column(Float, nullable=True)  # 0.0 to 1.0
    attempts = Column(Integer, default=0)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_session_module', 'session_id', 'module_id'),
    )


class PaperTradingPortfolio(Base):
    """User's paper trading portfolio metadata"""
    __tablename__ = 'portfolios'

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    portfolio_name = Column(String(200), default="My Portfolio")
    initial_cash = Column(Float, nullable=False)
    current_cash = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PaperTrade(Base):
    """Individual paper trades executed by users"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)
    portfolio_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    total_cost = Column(Float, nullable=False)  # Including commission
    strategy_used = Column(String(100), nullable=True)  # Optional strategy name
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index('idx_portfolio_timestamp', 'portfolio_id', 'timestamp'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )


class PortfolioSnapshot(Base):
    """Daily snapshots of portfolio value for performance tracking"""
    __tablename__ = 'portfolio_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String(100), unique=True, nullable=False)
    portfolio_id = Column(String(100), nullable=False, index=True)
    snapshot_date = Column(DateTime, nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    holdings = Column(Text, nullable=False)  # JSON: {symbol: {qty, avg_price, current_price}}
    metrics = Column(Text, nullable=True)  # JSON: {sharpe, max_dd, returns, etc.}
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_portfolio_date', 'portfolio_id', 'snapshot_date'),
    )


class MarketDataCache(Base):
    """Cache for market data from APIs to reduce API calls"""
    __tablename__ = 'market_data_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    data_type = Column(String(50), nullable=False)  # 'quote', 'profile', 'news', 'historical'
    data = Column(Text, nullable=False)  # JSON string
    fetched_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_symbol_type', 'symbol', 'data_type'),
        Index('idx_expires_at', 'expires_at'),
    )


class IPOWatchlist(Base):
    """User's IPO watchlist"""
    __tablename__ = 'ipo_watchlist'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    company_name = Column(String(200), nullable=False)
    symbol = Column(String(20), nullable=True)
    ipo_date = Column(DateTime, nullable=True)
    expected_valuation = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow)


# Database Management Functions

def init_database(database_url=None):
    """Initialize the database and create all tables"""
    if database_url is None:
        database_url = Config.get_database_url()

    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    """Get a database session"""
    if engine is None:
        engine = init_database()

    Session = sessionmaker(bind=engine)
    return Session()


def clear_expired_cache(session):
    """Remove expired cache entries"""
    now = datetime.utcnow()
    deleted = session.query(MarketDataCache).filter(
        MarketDataCache.expires_at < now
    ).delete()
    session.commit()
    return deleted


# Helper functions for common operations

def save_user_progress(session, session_id, module_id, completed, quiz_score=None):
    """Save or update user progress for a module"""
    progress = session.query(UserProgress).filter_by(
        session_id=session_id,
        module_id=module_id
    ).first()

    if progress:
        progress.completed = completed
        if quiz_score is not None:
            progress.quiz_score = quiz_score
        progress.attempts += 1
        if completed:
            progress.completed_at = datetime.utcnow()
        progress.updated_at = datetime.utcnow()
    else:
        progress = UserProgress(
            session_id=session_id,
            module_id=module_id,
            completed=completed,
            quiz_score=quiz_score,
            attempts=1,
            completed_at=datetime.utcnow() if completed else None
        )
        session.add(progress)

    session.commit()
    return progress


def get_user_progress(session, session_id):
    """Get all progress for a user"""
    return session.query(UserProgress).filter_by(session_id=session_id).all()


def cache_market_data(session, symbol, data_type, data, ttl_seconds):
    """Cache market data with TTL"""
    from datetime import timedelta

    # Delete existing cache for this symbol/type
    session.query(MarketDataCache).filter_by(
        symbol=symbol,
        data_type=data_type
    ).delete()

    # Create new cache entry
    cache_entry = MarketDataCache(
        symbol=symbol,
        data_type=data_type,
        data=json.dumps(data),
        fetched_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds)
    )
    session.add(cache_entry)
    session.commit()
    return cache_entry


def get_cached_data(session, symbol, data_type):
    """Retrieve cached data if not expired"""
    now = datetime.utcnow()
    cache = session.query(MarketDataCache).filter_by(
        symbol=symbol,
        data_type=data_type
    ).filter(
        MarketDataCache.expires_at > now
    ).first()

    if cache:
        return json.loads(cache.data)
    return None


if __name__ == "__main__":
    # Test database creation
    print("Initializing database...")
    engine = init_database()
    print(f"Database created at: {Config.DATABASE_PATH}")
    print("All tables created successfully!")

    # Test session
    session = get_session(engine)
    print(f"Database session created: {session}")
    session.close()
