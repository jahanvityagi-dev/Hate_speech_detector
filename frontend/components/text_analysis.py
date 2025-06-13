import streamlit as st
from services.api_client import APIClient
from services.validation import TextValidator, ValidationError
from services.session import SessionState
from core.models import ModerationResult


def render_text_analysis():
    st.markdown("<h2 class='section-header'>💬 Text Analysis</h2>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter text to analyze for hate speech",
        height=120,
        placeholder="Type your message here...",
        key="text_input"
    )

    if user_input:
        try:
            cleaned_text = TextValidator.validate_text(user_input)
            st.success("✅ Text is valid for analysis")
        except ValidationError as e:
            st.error(f"❌ {str(e)}")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        analyze_clicked = st.button(
            "🔍 Analyze Text",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing
        )
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            SessionState.clear_results()
            st.rerun()
    with col3:
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    return user_input, analyze_clicked
