import streamlit as st
from datetime import datetime

class SessionState:
    """Centralized session state management."""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables."""
        defaults = {
            "moderation_result": None,
            "processing": False,
            "last_input": "",
            "transcription_result": None,
            "analysis_count": 0,
            "last_analysis_time": None,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def clear_results():
        """Clear analysis results."""
        st.session_state.moderation_result = None
        st.session_state.transcription_result = None
    
    @staticmethod
    def increment_analysis_count():
        """Increment analysis counter."""
        st.session_state.analysis_count += 1
        st.session_state.last_analysis_time = datetime.now()