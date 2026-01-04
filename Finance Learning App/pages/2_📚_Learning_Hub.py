"""
Learning Hub - Interactive Learning Modules with Quizzes
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from components.quiz_engine import QuizEngine, load_learning_modules, render_module_card
from database.models import get_session, get_user_progress
import json

st.set_page_config(
    page_title="Learning Hub - Quantitative Finance Platform",
    page_icon="📚",
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
    <h1 style='margin: 0; color: white;'>📚 Learning Hub</h1>
    <p style='margin: 0.5rem 0 0 0; color: {Config.GS_GOLD}; font-size: 1.1rem;'>
        Master quantitative finance from fundamentals to advanced strategies
    </p>
</div>
""", unsafe_allow_html=True)

# Load modules
modules = load_learning_modules()

if not modules:
    st.error("No learning modules available. Please check the configuration.")
    st.stop()

# Get user progress
try:
    session = get_session()
    session_id = st.session_state.get(Config.SESSION_ID, 'default')
    progress_records = get_user_progress(session, session_id)

    # Convert to dictionary
    user_progress = {}
    for record in progress_records:
        user_progress[record.module_id] = {
            'completed': record.completed,
            'quiz_score': record.quiz_score,
            'attempts': record.attempts
        }
    session.close()
except Exception as e:
    st.warning(f"Could not load progress: {e}")
    user_progress = {}

# Calculate overall progress
total_modules = len(modules)
completed_modules = sum(1 for module_id in modules if user_progress.get(module_id, {}).get('completed', False))
progress_percentage = (completed_modules / total_modules * 100) if total_modules > 0 else 0

# Progress overview
st.subheader("📊 Your Progress")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Overall Progress", f"{progress_percentage:.0f}%", f"{completed_modules}/{total_modules} modules")

with col2:
    avg_score = 0
    if progress_records:
        scores = [r.quiz_score for r in progress_records if r.quiz_score is not None]
        avg_score = (sum(scores) / len(scores) * 100) if scores else 0
    st.metric("Average Score", f"{avg_score:.0f}%")

with col3:
    total_attempts = sum(r.attempts for r in progress_records)
    st.metric("Total Attempts", total_attempts)

with col4:
    pass_rate = (completed_modules / total_attempts * 100) if total_attempts > 0 else 0
    st.metric("Pass Rate", f"{pass_rate:.0f}%")

# Progress bar
st.progress(progress_percentage / 100)

st.markdown("---")

# Module selection interface
st.subheader("🎓 Learning Modules")

# Tab selection
tab1, tab2, tab3 = st.tabs(["📋 All Modules", "🎯 In Progress", "✅ Completed"])

with tab1:
    st.markdown("### All Available Modules")

    # Group by category
    beginner_modules = {k: v for k, v in modules.items() if v.get('category') == 'beginner'}
    intermediate_modules = {k: v for k, v in modules.items() if v.get('category') == 'intermediate'}
    advanced_modules = {k: v for k, v in modules.items() if v.get('category') == 'advanced'}

    if beginner_modules:
        st.markdown("#### 🟢 Beginner Modules")
        for module_id, module in beginner_modules.items():
            render_module_card(module_id, module, user_progress)

            if st.button(f"Start: {module['title']}", key=f"start_{module_id}"):
                st.session_state['selected_module'] = module_id
                st.rerun()

    if intermediate_modules:
        st.markdown("#### 🟡 Intermediate Modules")
        for module_id, module in intermediate_modules.items():
            render_module_card(module_id, module, user_progress)

            if st.button(f"Start: {module['title']}", key=f"start_{module_id}"):
                st.session_state['selected_module'] = module_id
                st.rerun()

    if advanced_modules:
        st.markdown("#### 🔴 Advanced Modules")
        for module_id, module in advanced_modules.items():
            render_module_card(module_id, module, user_progress)

            if st.button(f"Start: {module['title']}", key=f"start_{module_id}"):
                st.session_state['selected_module'] = module_id
                st.rerun()

with tab2:
    st.markdown("### Modules In Progress")
    in_progress = {k: v for k, v in modules.items()
                   if k in user_progress and not user_progress[k].get('completed', False)}

    if in_progress:
        for module_id, module in in_progress.items():
            render_module_card(module_id, module, user_progress)

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Continue", key=f"continue_{module_id}"):
                    st.session_state['selected_module'] = module_id
                    st.rerun()
    else:
        st.info("No modules in progress. Start learning from the 'All Modules' tab!")

with tab3:
    st.markdown("### Completed Modules")
    completed = {k: v for k, v in modules.items()
                 if user_progress.get(k, {}).get('completed', False)}

    if completed:
        for module_id, module in completed.items():
            render_module_card(module_id, module, user_progress)

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Review", key=f"review_{module_id}"):
                    st.session_state['selected_module'] = module_id
                    st.rerun()
    else:
        st.info("Complete your first module to see it here!")

st.markdown("---")

# Module detail view
if 'selected_module' in st.session_state and st.session_state['selected_module']:
    module_id = st.session_state['selected_module']

    if module_id in modules:
        module = modules[module_id]

        # Back button
        if st.button("← Back to Module List"):
            del st.session_state['selected_module']
            st.rerun()

        st.markdown("---")

        # Module header
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {Config.GS_BLUE} 0%, {Config.GS_DARK_BLUE} 100%);
                    color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;'>
            <h2 style='margin: 0; color: white;'>{module['title']}</h2>
            <p style='margin: 0.5rem 0 0 0; color: {Config.GS_GOLD};'>
                Difficulty {module['difficulty']}/5 • {module['estimated_time']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Prerequisites check
        if module.get('prerequisites'):
            st.info(f"**Prerequisites:** {', '.join([modules[p]['title'] for p in module['prerequisites'] if p in modules])}")

        # Module content
        st.markdown("## 📖 Learning Content")
        st.markdown(module['content'])

        st.markdown("---")

        # Quiz
        if 'quiz' in module and module['quiz']:
            quiz_engine = QuizEngine(module_id, module['quiz'])
            quiz_engine.render_quiz()
        else:
            st.warning("No quiz available for this module.")
    else:
        st.error("Module not found!")
        del st.session_state['selected_module']

else:
    # Welcome message when no module selected
    st.info("""
    💡 **How to Use the Learning Hub:**

    1. **Browse Modules**: Explore modules in the tabs above
    2. **Start Learning**: Click on a module to begin
    3. **Complete Quiz**: Test your knowledge at the end
    4. **Track Progress**: Monitor your advancement in the dashboard
    5. **Unlock Advanced Content**: Pass beginner modules to access advanced topics

    Start with the beginner modules if you're new to quantitative finance!
    """)

    # Quick start recommendations
    st.markdown("### 🚀 Recommended Starting Path")

    beginner_path = [
        ("intro_to_stocks", "Introduction to Stock Markets"),
        ("risk_and_return", "Understanding Risk and Return"),
        ("technical_analysis", "Technical Analysis Fundamentals")
    ]

    for i, (mod_id, title) in enumerate(beginner_path, 1):
        if mod_id in modules:
            completed = user_progress.get(mod_id, {}).get('completed', False)
            status = "✅" if completed else "⭕"

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"{status} **{i}. {title}**")
            with col2:
                if not completed:
                    if st.button("Start", key=f"quickstart_{mod_id}"):
                        st.session_state['selected_module'] = mod_id
                        st.rerun()

# Footer
st.markdown("---")
st.caption("""
**Learning Tip:** Take your time with each module. Understanding the concepts is more important than speed.
Master the basics before moving to advanced topics. You can retake quizzes as many times as needed.
""")
