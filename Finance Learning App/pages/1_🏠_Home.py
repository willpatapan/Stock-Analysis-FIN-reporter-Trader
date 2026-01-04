"""
Home Dashboard - Overview and Quick Access
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config

st.set_page_config(
    page_title="Home - Quantitative Finance Platform",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Dashboard Overview")

st.markdown(f"""
<div style='background: linear-gradient(135deg, {Config.GS_BLUE} 0%, {Config.GS_DARK_BLUE} 100%);
            color: white; padding: 2rem; border-radius: 8px; margin-bottom: 2rem;
            border-bottom: 3px solid {Config.GS_GOLD};'>
    <h1 style='margin: 0; color: white;'>Welcome to Your Quantitative Finance Platform</h1>
    <p style='margin: 0.5rem 0 0 0; color: {Config.GS_GOLD}; font-size: 1.1rem;'>
        Master trading strategies used by top Wall Street firms
    </p>
</div>
""", unsafe_allow_html=True)

st.info("""
**🎓 Phase 2 Complete: Learning Hub & Classic Strategies**

Your platform includes:
- 📚 **Learning Hub**: 3 comprehensive modules with interactive quizzes
- 📈 **Classic Strategies**: Backtest 4 proven strategies (Value, Growth, Momentum, Mean Reversion)
- 📊 **Live Data**: Real market data for accurate backtesting
- 💾 **Progress Tracking**: Your learning progress is automatically saved

Use the sidebar to navigate between sections.
""")

# Quick stats
st.subheader("📊 Quick Stats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Learning Progress", "0%", "Start learning!")

with col2:
    st.metric("Paper Portfolio", "$100,000", "Initial capital")

with col3:
    st.metric("Modules Completed", "0/10", "Begin your journey")

with col4:
    st.metric("Trading Days", "0", "Start trading")

st.markdown("---")

# Quick access buttons
st.subheader("🚀 Quick Access")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📚 Start Learning", use_container_width=True, type="primary"):
        st.switch_page("pages/2_📚_Learning_Hub.py")

with col2:
    if st.button("📈 Backtest Strategies", use_container_width=True, type="primary"):
        st.switch_page("pages/3_📈_Classic_Strategies.py")

with col3:
    st.info("🔜 **Coming Soon**\n\nPaper Trading & Portfolio Analysis")

st.markdown("---")

# Feature overview
st.subheader("📋 Available Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Ready Now (Phase 2)

    **📚 Learning Hub**
    - Introduction to Stock Markets
    - Understanding Risk and Return
    - Technical Analysis Fundamentals
    - Interactive quizzes with explanations

    **📈 Classic Strategies**
    - 💰 Value Investing (Warren Buffett style)
    - 🚀 Growth Investing (Peter Lynch style)
    - 📊 Momentum Trading (Trend following)
    - 🔄 Mean Reversion (Statistical trading)
    - Full backtesting with performance metrics
    """)

with col2:
    st.markdown("""
    ### 🔜 Coming Soon (Phase 3+)

    **🧮 Advanced Quant**
    - Black-Scholes options pricing
    - Portfolio optimization (MPT)
    - Pairs trading analyzer

    **🎯 Paper Trading**
    - Live trading simulator
    - Real-time portfolio tracking

    **💼 Investment Banking**
    - M&A accretion/dilution models
    - DCF valuation tools
    """)

st.success("""
✅ **Platform Status: Phase 2 Complete - All Systems Operational**

Start with the **📚 Learning Hub** to build your foundation, then practice with **📈 Classic Strategies** backtesting!
""")
