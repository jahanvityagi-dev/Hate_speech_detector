import streamlit as st
def render_header():
        """Render the main application header."""
        st.markdown(
            '<h1 class="main-header">🛡️ Hate Speech Detection Assistant</h1>', 
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="sub-header">Advanced AI-powered content moderation and policy enforcement</p>', 
            unsafe_allow_html=True
        )