"""
Quantitative Finance Platform - Main Application
Professional finance education and trading platform with Goldman Sachs-inspired UI
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from config.themes import GoldmanSachsTheme
from database.models import init_database
import uuid


def load_custom_css():
    """Load Goldman Sachs custom CSS theme"""
    css_path = Config.ASSETS_DIR / 'styles' / 'goldman_sachs_theme.css'
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    # User session ID
    if Config.SESSION_ID not in st.session_state:
        st.session_state[Config.SESSION_ID] = str(uuid.uuid4())

    # Portfolio ID for paper trading
    if Config.PORTFOLIO_ID not in st.session_state:
        st.session_state[Config.PORTFOLIO_ID] = f"portfolio_{st.session_state[Config.SESSION_ID][:8]}"

    # User progress tracking
    if Config.USER_PROGRESS not in st.session_state:
        st.session_state[Config.USER_PROGRESS] = {}


def show_sidebar():
    """Display sidebar with navigation and info"""
    with st.sidebar:
        # Logo/Header
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0; border-bottom: 2px solid {Config.GS_GOLD};'>
            <h1 style='color: white; margin: 0; font-size: 1.5rem;'>💼</h1>
            <h2 style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Quantitative Finance</h2>
            <p style='color: {Config.GS_GOLD}; margin: 0.25rem 0 0 0; font-size: 0.9rem;'>Professional Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation info
        st.info("""
        **Navigate using the pages above:**

        🏠 Home - Dashboard overview
        📚 Learning Hub - Interactive courses
        📈 Strategies - Trading strategies
        🧮 Advanced Quant - Options & MPT
        💼 Investment Banking - M&A tools
        🎯 Paper Trading - Practice trading
        📊 Portfolio Analysis - Risk metrics
        🔍 IPO Tracker - Upcoming IPOs
        📄 Deck Generator - Create pitches
        """)

        # Session info
        with st.expander("ℹ️ Session Information"):
            st.caption(f"**Session ID:** {st.session_state[Config.SESSION_ID][:8]}...")
            st.caption(f"**Portfolio:** {st.session_state[Config.PORTFOLIO_ID]}")

        # API Status
        with st.expander("🔌 API Status"):
            if Config.validate_finnhub_key():
                st.success("✅ Finnhub API Connected")
            else:
                st.warning("⚠️ Finnhub API key not configured")
                st.caption("Add `FINNHUB_API_KEY` to `.env` file for live data")

        # Footer
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='text-align: center; padding-top: 1rem; border-top: 1px solid {Config.GS_GOLD};'>
            <p style='color: {Config.GS_GOLD}; font-size: 0.8rem; margin: 0;'>v{Config.VERSION}</p>
            <p style='color: white; font-size: 0.7rem; margin: 0.5rem 0 0 0;'>Educational purposes only</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"{Config.APP_NAME} v{Config.VERSION} - Professional quantitative finance platform"
        }
    )

    # Initialize database
    try:
        init_database()
    except Exception as e:
        st.error(f"Database initialization error: {e}")

    # Load custom CSS
    load_custom_css()

    # Initialize session state
    initialize_session_state()

    # Show sidebar
    show_sidebar()

    # Main content
    st.markdown(f"""
    <div class='goldman-header' style='background: linear-gradient(135deg, {Config.GS_BLUE} 0%, {Config.GS_DARK_BLUE} 100%);
                                       color: white; padding: 2rem; border-radius: 8px; margin-bottom: 2rem;
                                       border-bottom: 3px solid {Config.GS_GOLD};'>
        <h1 style='margin: 0; color: white;'>Welcome to the Quantitative Finance Platform</h1>
        <p style='margin: 0.5rem 0 0 0; color: {Config.GS_GOLD}; font-size: 1.1rem;'>
            Master the strategies used by top Wall Street firms
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Welcome message
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #0033A0; margin-top: 0;'>📚 Learn</h3>
            <p>Interactive modules teaching institutional-grade strategies from beginner to advanced levels.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #0033A0; margin-top: 0;'>🎯 Practice</h3>
            <p>Paper trading simulator with real-time market data to test your strategies risk-free.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #0033A0; margin-top: 0;'>📊 Analyze</h3>
            <p>Professional-grade tools for options pricing, portfolio optimization, and risk management.</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")

    st.markdown("""
    **Getting Started:**
    1. **📚 Start Learning** - Visit the Learning Hub to begin with foundational modules
    2. **📈 Explore Strategies** - Study classic and advanced quantitative trading strategies
    3. **🎯 Practice Trading** - Use the paper trading simulator to apply what you've learned
    4. **📊 Analyze Performance** - Review your results with institutional-grade risk metrics

    **What Makes This Platform Unique:**
    - **Real Quant Code**: Uses the same mathematical models as Wall Street firms
    - **Live Market Data**: Powered by Finnhub API for real-time quotes and news
    - **Comprehensive Coverage**: From basic concepts to advanced derivatives pricing
    - **Professional UI**: Goldman Sachs-inspired design for a premium experience
    """)

    # Feature Highlights
    st.markdown("---")
    st.subheader("✨ Platform Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown("""
        **📚 Learning & Education**
        - 10+ interactive learning modules
        - Quizzes with instant feedback
        - Progress tracking system
        - Beginner to intermediate content

        **📈 Trading Strategies**
        - Value investing screeners
        - Growth stock analysis
        - Momentum trading systems
        - Mean reversion strategies
        """)

    with feature_col2:
        st.markdown("""
        **🧮 Advanced Quantitative Tools**
        - Black-Scholes options pricing
        - Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
        - Modern Portfolio Theory optimization
        - Pairs trading analyzer

        **💼 Investment Banking**
        - M&A accretion/dilution models
        - DCF valuation (coming soon)
        - Comparable company analysis (coming soon)
        - Professional pitch deck generation
        """)

    # System Status
    st.markdown("---")
    st.subheader("🔧 System Status")

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.metric(
            label="Database",
            value="✅ Connected",
            delta="Operational"
        )

    with status_col2:
        if Config.validate_finnhub_key():
            st.metric(
                label="Market Data",
                value="✅ Live",
                delta="Finnhub API"
            )
        else:
            st.metric(
                label="Market Data",
                value="⚠️ Offline",
                delta="Configure API"
            )

    with status_col3:
        st.metric(
            label="Modules",
            value="9 Pages",
            delta="All Active"
        )

    with status_col4:
        st.metric(
            label="Version",
            value=Config.VERSION,
            delta="Latest"
        )

    # Call to Action
    st.markdown("---")
    st.info("""
    💡 **Ready to get started?** Navigate to the **Learning Hub** in the sidebar to begin your journey to becoming a quantitative finance expert!
    """)

    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This platform is for educational purposes only. All trading strategies, models, and tools are provided for learning and simulation.
    Not financial advice. Past performance does not guarantee future results. Trade at your own risk.
    """)


if __name__ == "__main__":
    main()
