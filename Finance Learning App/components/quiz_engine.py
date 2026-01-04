"""
Quiz Engine Component
Interactive quiz system with scoring and progress tracking
"""

import streamlit as st
import json
from typing import Dict, List, Any
from datetime import datetime
from database.models import save_user_progress, get_session
from config.settings import Config


class QuizEngine:
    """
    Interactive quiz system for learning modules

    Features:
    - Multiple choice questions
    - True/False questions
    - Immediate feedback with explanations
    - Score calculation
    - Progress tracking
    """

    def __init__(self, module_id: str, questions: List[Dict]):
        """
        Initialize quiz engine

        Args:
            module_id: Unique identifier for the learning module
            questions: List of question dictionaries
        """
        self.module_id = module_id
        self.questions = questions
        self.total_questions = len(questions)

        # Initialize session state for this quiz
        if f'quiz_{module_id}_answers' not in st.session_state:
            st.session_state[f'quiz_{module_id}_answers'] = {}
        if f'quiz_{module_id}_submitted' not in st.session_state:
            st.session_state[f'quiz_{module_id}_submitted'] = False
        if f'quiz_{module_id}_score' not in st.session_state:
            st.session_state[f'quiz_{module_id}_score'] = 0

    def render_question(self, question_idx: int, question: Dict):
        """
        Render a single question

        Args:
            question_idx: Index of the question
            question: Question dictionary
        """
        st.markdown(f"### Question {question_idx + 1} of {self.total_questions}")
        st.markdown(f"**{question['question']}**")

        question_type = question['type']
        question_key = f"q_{self.module_id}_{question_idx}"

        # Get current answer from session state
        current_answer = st.session_state[f'quiz_{self.module_id}_answers'].get(question_idx)

        if question_type == 'multiple_choice':
            # Multiple choice question
            selected = st.radio(
                "Select your answer:",
                options=range(len(question['options'])),
                format_func=lambda x: question['options'][x],
                key=question_key,
                index=current_answer if current_answer is not None else None,
                disabled=st.session_state[f'quiz_{self.module_id}_submitted']
            )

            if selected is not None:
                st.session_state[f'quiz_{self.module_id}_answers'][question_idx] = selected

        elif question_type == 'true_false':
            # True/False question
            selected = st.radio(
                "Select your answer:",
                options=[True, False],
                format_func=lambda x: "True" if x else "False",
                key=question_key,
                index=0 if current_answer is True else (1 if current_answer is False else None),
                disabled=st.session_state[f'quiz_{self.module_id}_submitted']
            )

            if selected is not None:
                st.session_state[f'quiz_{self.module_id}_answers'][question_idx] = selected

        # Show feedback if quiz is submitted
        if st.session_state[f'quiz_{self.module_id}_submitted']:
            self._show_feedback(question_idx, question)

        st.markdown("---")

    def _show_feedback(self, question_idx: int, question: Dict):
        """Show feedback for a question after submission"""
        user_answer = st.session_state[f'quiz_{self.module_id}_answers'].get(question_idx)
        correct_answer = question['correct']

        if user_answer == correct_answer:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect. The correct answer was: **{self._format_correct_answer(question)}**")

        # Show explanation
        if 'explanation' in question:
            st.info(f"**Explanation:** {question['explanation']}")

    def _format_correct_answer(self, question: Dict) -> str:
        """Format the correct answer for display"""
        if question['type'] == 'multiple_choice':
            return question['options'][question['correct']]
        elif question['type'] == 'true_false':
            return "True" if question['correct'] else "False"
        return ""

    def calculate_score(self) -> tuple:
        """
        Calculate quiz score

        Returns:
            Tuple of (score_percentage, correct_count, total_count)
        """
        correct = 0
        total = len(self.questions)

        for idx, question in enumerate(self.questions):
            user_answer = st.session_state[f'quiz_{self.module_id}_answers'].get(idx)
            if user_answer == question['correct']:
                correct += 1

        score_percentage = (correct / total) * 100 if total > 0 else 0
        return score_percentage, correct, total

    def render_quiz(self):
        """Render the complete quiz"""
        st.subheader("📝 Quiz")

        # Render all questions
        for idx, question in enumerate(self.questions):
            self.render_question(idx, question)

        # Submit button
        col1, col2 = st.columns([1, 4])

        with col1:
            if not st.session_state[f'quiz_{self.module_id}_submitted']:
                if st.button("Submit Quiz", type="primary", use_container_width=True):
                    # Check if all questions answered
                    unanswered = []
                    for idx in range(len(self.questions)):
                        if idx not in st.session_state[f'quiz_{self.module_id}_answers']:
                            unanswered.append(idx + 1)

                    if unanswered:
                        st.error(f"⚠️ Please answer all questions. Missing: {', '.join(map(str, unanswered))}")
                    else:
                        st.session_state[f'quiz_{self.module_id}_submitted'] = True
                        score, correct, total = self.calculate_score()
                        st.session_state[f'quiz_{self.module_id}_score'] = score

                        # Save progress to database
                        try:
                            session = get_session()
                            session_id = st.session_state.get(Config.SESSION_ID, 'default')
                            passed = score >= (Config.QUIZ_PASS_THRESHOLD * 100)

                            save_user_progress(
                                session,
                                session_id,
                                self.module_id,
                                passed,
                                score / 100  # Store as decimal
                            )
                            session.close()
                        except Exception as e:
                            st.error(f"Error saving progress: {e}")

                        st.rerun()
            else:
                if st.button("Retake Quiz", use_container_width=True):
                    # Reset quiz
                    st.session_state[f'quiz_{self.module_id}_answers'] = {}
                    st.session_state[f'quiz_{self.module_id}_submitted'] = False
                    st.session_state[f'quiz_{self.module_id}_score'] = 0
                    st.rerun()

        # Show results if submitted
        if st.session_state[f'quiz_{self.module_id}_submitted']:
            st.markdown("---")
            score, correct, total = self.calculate_score()

            st.subheader("📊 Quiz Results")

            # Score display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Score", f"{score:.1f}%")

            with col2:
                st.metric("Correct", f"{correct}/{total}")

            with col3:
                pass_threshold = Config.QUIZ_PASS_THRESHOLD * 100
                if score >= pass_threshold:
                    st.metric("Status", "✅ Passed")
                else:
                    st.metric("Status", "❌ Failed")

            # Pass/Fail message
            if score >= pass_threshold:
                st.success(f"""
                🎉 **Congratulations!** You passed with {score:.1f}%!

                You've demonstrated mastery of this module and can now proceed to the next one.
                """)
            else:
                st.warning(f"""
                📚 **Keep Learning!** You scored {score:.1f}%, but need {pass_threshold:.0f}% to pass.

                Review the material and try again. You can retake the quiz as many times as needed.
                """)

    def get_progress_summary(self) -> Dict:
        """Get summary of quiz progress"""
        submitted = st.session_state.get(f'quiz_{self.module_id}_submitted', False)
        score = st.session_state.get(f'quiz_{self.module_id}_score', 0)

        return {
            'module_id': self.module_id,
            'submitted': submitted,
            'score': score,
            'passed': score >= (Config.QUIZ_PASS_THRESHOLD * 100) if submitted else False,
            'total_questions': self.total_questions
        }


def load_learning_modules() -> Dict:
    """Load learning modules from JSON file"""
    try:
        modules_path = Config.BASE_DIR / 'config' / 'learning_modules.json'
        with open(modules_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Learning modules file not found!")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Error parsing learning modules: {e}")
        return {}


def render_module_card(module_id: str, module: Dict, user_progress: Dict = None):
    """
    Render a learning module card

    Args:
        module_id: Module identifier
        module: Module data dictionary
        user_progress: Optional user progress data
    """
    # Check if completed
    completed = False
    score = 0
    if user_progress and module_id in user_progress:
        completed = user_progress[module_id].get('completed', False)
        score = user_progress[module_id].get('quiz_score', 0) * 100

    # Difficulty badge color
    difficulty_colors = {
        1: "🟢",
        2: "🟡",
        3: "🟠",
        4: "🔴",
        5: "🔴"
    }

    difficulty_badge = difficulty_colors.get(module['difficulty'], "⚪")

    # Status badge
    if completed:
        status_badge = f"✅ Completed ({score:.0f}%)"
        border_color = "#2ECC71"
    else:
        status_badge = "📚 Not Started"
        border_color = "#0033A0"

    st.markdown(f"""
    <div style='border-left: 4px solid {border_color}; padding: 1rem; background: white;
                border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
        <div style='display: flex; justify-content: space-between; align-items: start;'>
            <div style='flex: 1;'>
                <h3 style='margin: 0 0 0.5rem 0; color: #0033A0;'>{module['title']}</h3>
                <p style='margin: 0 0 0.5rem 0; color: #666;'>{module['description']}</p>
                <div style='display: flex; gap: 1rem; font-size: 0.9rem;'>
                    <span>{difficulty_badge} Difficulty {module['difficulty']}/5</span>
                    <span>⏱️ {module['estimated_time']}</span>
                    <span>{status_badge}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
