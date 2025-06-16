import streamlit as st
import logging
from frontend.services.session import SessionState
from frontend.components.styles import load_custom_styles
from frontend.components.header import render_header
from frontend.components.stats import render_stats_cards
from frontend.components.text_analysis import render_text_analysis
from frontend.components.audio_analysis import render_audio_analysis
from frontend.components.results import render_results
from frontend.components.sidebar import render_sidebar_info
from frontend.core.models import ModerationResult
from frontend.services.api_client import APIClient

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class ModerationApp:
    """Main class to orchestrate the moderation Streamlit app."""

    def __init__(self):
        self._setup_page()
        SessionState.initialize()
        load_custom_styles()

    def _setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Hate Speech Detection Assistant",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def _process_text(self, text: str) -> bool:
        """Handles text moderation pipeline."""
        try:
            st.session_state.processing = True
            with st.spinner("🔍 Analyzing text..."):
                result = APIClient.moderate_text(text)
                st.session_state.moderation_result = result.__dict__
                st.session_state.transcription_result = None
                SessionState.increment_analysis_count()
            return True
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Text processing error: {e}")
            return False
        finally:
            st.session_state.processing = False

    def _process_audio(self, audio_file) -> bool:
        """Handles audio moderation pipeline."""
        try:
            st.session_state.processing = True
            with st.spinner("🎵 Transcribing and analyzing audio..."):
                result, transcription = APIClient.transcribe_and_moderate(audio_file)
                st.session_state.moderation_result = result.__dict__
                st.session_state.transcription_result = transcription
                SessionState.increment_analysis_count()
            return True
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Audio processing error: {e}")
            return False
        finally:
            st.session_state.processing = False

    def run(self):
        """Main run loop of the app."""
        render_header()
        render_stats_cards()
        st.divider()

        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            text_input, analyze_text = render_text_analysis()
            st.divider()
            audio_file, analyze_audio = render_audio_analysis()

            if analyze_text and text_input:
                if self._process_text(text_input):
                    st.rerun()

            if analyze_audio and audio_file:
                if self._process_audio(audio_file):
                    st.rerun()

        with col2:
            render_sidebar_info()

            if st.session_state.moderation_result:
                result_data = st.session_state.moderation_result
                result = ModerationResult(**result_data)
                render_results(result, st.session_state.transcription_result)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: var(--text-secondary);">
                    <h3>👈 Ready for Analysis</h3>
                    <p>Enter text or upload audio to begin content analysis.</p>
                </div>
                """, unsafe_allow_html=True)

# Entry point
def main():
    try:
        app = ModerationApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        logger.critical(f"Critical error: {e}")
        if st.button("🚨 Emergency Reset"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
