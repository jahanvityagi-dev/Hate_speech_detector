import io
import os
import tempfile
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from config import Config
from services.session import SessionState
from services.api_client import APIClient
from services.validation import AudioValidator, ValidationError
import logging

logger = logging.getLogger(__name__)

def render_audio_analysis():
    st.markdown('<h2 class="section-header">🎤 Audio Analysis</h2>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="audio-section">', unsafe_allow_html=True)

        st.info("📝 Upload an audio file or record directly using your microphone. The audio will be transcribed and analyzed for hate speech.")

        st.markdown("#### Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=Config.ALLOWED_AUDIO_FORMATS,
            help=f"Supported formats: {', '.join(Config.ALLOWED_AUDIO_FORMATS)}. Max size: {Config.MAX_FILE_SIZE_MB}MB"
        )

        analyze_upload_clicked = False
        if uploaded_file:
            try:
                AudioValidator.validate_audio_file(uploaded_file)
                st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')

                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.success(f"✅ File validated: {file_size_mb:.1f}MB")

                analyze_upload_clicked = st.button(
                    "🎵 Transcribe & Analyze Upload", 
                    type="primary",
                    disabled=st.session_state.processing
                )

            except ValidationError as e:
                st.error(f"❌ {str(e)}")

        st.divider()

        st.markdown("#### Record Audio")
        with st.form("record_audio_form"):
            audio_bytes = audio_recorder(
                text="🎙️ Click to Record",
                recording_color="#ef4444",
                neutral_color="#6b7280",
                icon_size="2x"
            )
            record_submit = st.form_submit_button("🎵 Transcribe & Analyze Recording")

            st.write("🎧 Raw audio data:", "Available" if audio_bytes else "None")


        if record_submit and audio_bytes:

            st.audio(audio_bytes, format="audio/wav")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                with open(temp_path, "rb") as f:
                    audio_data = f.read()

                audio_buffer = io.BytesIO(audio_data)
                audio_buffer.name = "recorded.wav"

                result, transcription = APIClient.transcribe_and_moderate(audio_buffer)

                st.session_state.moderation_result = result.__dict__
                st.session_state.transcription_result = transcription
                SessionState.increment_analysis_count()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                logger.error(f"Audio processing error: {e}")

            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        st.markdown('</div>', unsafe_allow_html=True)

    return uploaded_file, analyze_upload_clicked
