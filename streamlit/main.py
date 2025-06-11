import streamlit as st
import requests
import json
import time
import tempfile
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import base64
from io import BytesIO

# ================================
# CONFIGURATION & CONSTANTS
# ================================

BACKEND_URL = "http://localhost:8000"
MODERATE_ENDPOINT = f"{BACKEND_URL}/moderate"
TRANSCRIBE_ENDPOINT = f"{BACKEND_URL}/transcribe-audio"

# Classification color mapping
CLASSIFICATION_COLORS = {
    "Hate": "#ff4444",      # Red
    "Toxic": "#ff8800",     # Orange
    "Offensive": "#ffdd00", # Yellow
    "Neutral": "#44aa44",   # Green
    "Ambiguous": "#888888"  # Gray
}



# ================================
# CUSTOM CSS STYLING
# ================================

def load_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    /* Main app styling */
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Classification badges */
    .classification-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Results container */
    .results-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    /* Policy snippets */
    .policy-snippet {
        background-color: #ffffff;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        color: #2c3e50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .policy-snippet strong {
        color: #1e7e34;
        font-weight: 600;
    }
    
    /* Action recommendation */
    .action-recommendation {
        background-color: #ffffff;
        border: 2px solid #fd7e14;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #fd7e14;
        color: #495057;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .action-recommendation strong {
        color: #dc6545;
        font-weight: 600;
    }
    
    /* Audio section */
    .audio-section {
        background-color: #f1f3f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ================================
# UTILITY FUNCTIONS
# ================================

def initialize_session_state():
    """Initialize session state variables."""
    if "moderation_result" not in st.session_state:
        st.session_state.moderation_result = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_input" not in st.session_state:
        st.session_state.last_input = ""
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None

def validate_text_input(text: str) -> Tuple[bool, str]:
    """
    Validate text input according to requirements.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    text_length = len(text.strip())
    if text_length < 10:
        return False, f"Text must be at least 10 characters long. Current: {text_length}"
    
    if text_length > 1000:
        return False, f"Text must be no more than 1000 characters long. Current: {text_length}"
    
    return True, ""

def make_api_request(endpoint: str, data: Dict[str, Any] = None, files: Dict = None) -> Optional[Dict[str, Any]]:
    """
    Make API request to backend with proper error handling.
    
    Args:
        endpoint: API endpoint URL
        data: JSON data for POST request
        files: Files for multipart upload
        
    Returns:
        API response as dictionary or None if failed
    """
    try:
        if files:
            response = requests.post(endpoint, files=files, timeout=30)
        else:
            response = requests.post(endpoint, json=data, timeout=30)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please ensure the API is running at http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("❌ Invalid response from server.")
        return None

def create_classification_badge(classification: str) -> str:
    """Create HTML for classification badge with appropriate color."""
    color = CLASSIFICATION_COLORS.get(classification, "#888888")
    return f"""
    <div class="classification-badge" style="background-color: {color};">
        {classification}
    </div>
    """

def export_results_as_text(result: Dict[str, Any], transcription: str = None) -> str:
    """Export moderation results as formatted text matching backend structure."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    export_text = f"""
HATE SPEECH DETECTION REPORT
Generated: {timestamp}
========================================

INPUT TEXT:
{result.get('input', 'N/A')}
"""
    
    if transcription:
        export_text += f"""
TRANSCRIPTION:
{transcription}
"""
    
    export_text += f"""
CLASSIFICATION LABEL: {result.get('label', 'N/A')}

CLASSIFICATION REASON:
{result.get('classification_reason', 'N/A')}

DETAILED EXPLANATION:
{result.get('explanation', 'N/A')}

RELEVANT POLICY SNIPPETS:
"""
    
    policy_snippets = result.get('policy_snippets', [])
    for i, policy in enumerate(policy_snippets, 1):
        export_text += f"""
{i}. Source: {policy.get('source_file', 'Unknown')}
   Content: {policy.get('text', 'N/A')}
   Relevance Score: {policy.get('score', 'N/A')}
"""
    
    export_text += f"""
RECOMMENDED ACTION: {result.get('action', 'N/A')}

ACTION REASON:
{result.get('action_reason', 'N/A')}

========================================
Report generated by Hate Speech Detection System
"""
    
    return export_text


# ================================
# ADDITION TO AUDIO SECTION
# ================================

def render_audio_section():
    """Render the audio recording section."""
    st.markdown("## 🎤 Audio Analysis")
    st.markdown('<div class="audio-section">', unsafe_allow_html=True)

    st.info("You can either upload an audio file or record using your mic.")

    audio = audio_recorder("🎙️ Start Recording", "Stop Recording")
    analyze_audio_clicked = False

    if audio:
        st.audio(audio.tobytes(), format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio.tobytes())
            temp_path = f.name

        with open(temp_path, "rb") as f:
            files = {"file": f}
            response = requests.post(TRANSCRIBE_ENDPOINT, files=files)

        if response.status_code == 200:
            result = response.json()
            transcription = result.get("transcribed_text", "")
            moderation_result = result.get("moderation_result", {})
            st.session_state.moderation_result = moderation_result
            st.session_state.transcription_result = transcription
            st.rerun()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    st.markdown("</div>", unsafe_allow_html=True)
    return None, analyze_audio_clicked

# ================================
# UI COMPONENTS
# ================================

def render_header():
    """Render the main header section."""
    st.markdown('<h1 class="main-header">🛡️ Hate Speech Detection Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered content moderation and policy enforcement</p>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with information and legend."""
    with st.sidebar:
        st.markdown("## 📋 About This Tool")
        st.markdown("""
        <div class="sidebar-info">
        This system uses advanced AI to detect and classify potentially harmful content,
        providing policy-based explanations and moderation recommendations.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 🏷️ Classification Types")
        for classification, color in CLASSIFICATION_COLORS.items():
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div style="width: 20px; height: 20px; background-color: {color}; 
                           border-radius: 50%; margin-right: 0.5rem;"></div>
                <strong>{classification}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("## ⚠️ Disclaimer")
        st.markdown("""
        <div class="sidebar-info">
        This tool is for educational and demonstration purposes. 
        Always review AI decisions critically and consider human oversight 
        for production moderation systems.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 📊 Usage Statistics")
        if st.session_state.moderation_result:
            st.success("✅ Last analysis completed")
        else:
            st.info("💭 Ready for analysis")

def render_text_input_section():
    """Render the text input section."""
    st.markdown("## 💬 Text Analysis")
    
    user_input = st.text_area(
        "Enter text to analyze for hate speech",
        height=120,
        placeholder="Type your message here... ",
        help="Enter any text content you'd like to analyze for potential policy violations."
    )
    
    # Input validation display
    if user_input:
        char_count = len(user_input.strip())
        # if char_count < 10:
        #     st.warning(f"⚠️ Text too short: {char_count}/10 minimum characters")
        if char_count > 1000:
            st.error(f"❌ Text too long: {char_count}/1000 maximum characters")
        else:
            st.success(f"✅ Text length: {char_count} characters")
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analyze_clicked = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.moderation_result = None
            st.session_state.transcription_result = None
            st.rerun()
    
    with col3:
        st.button("🔄 Reset", use_container_width=True)
    
    return user_input, analyze_clicked

def render_audio_section():
    """Render the audio recording section."""
    st.markdown("## 🎤 Audio Analysis")
    st.markdown('<div class="audio-section">', unsafe_allow_html=True)
    
    st.info(" You can either upload an audio file or record using your microphone. " +
            "Upload an audio file or use the text input above.")
    
    # File upload for audio
    uploaded_file = st.file_uploader(
        "Upload audio file (WAV/MP3/OGG/M4A):",
        type=['wav', 'mp3', 'ogg', 'm4a'],
        help="Upload an audio file to transcribe and analyze for hate speech."
    )
    
    analyze_audio_clicked = False
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
        analyze_audio_clicked = st.button("🎵 Transcribe & Analyze Audio", type="primary")
    
    st.markdown("---")

    # === 🎙️ Audio Recorder Button ===
    st.markdown("#### record audio now:")
    audio_bytes = audio_recorder(text="🎙️ Click to Record", icon_size="2x")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        with open(temp_path, "rb") as f:
            files = {"file": f}
            response = requests.post(TRANSCRIBE_ENDPOINT, files=files)

        if response.status_code == 200:
            result = response.json()
            transcription = result.get("transcribed_text", "")
            moderation_result = result.get("moderation_result", {})
            st.session_state.moderation_result = moderation_result
            st.session_state.transcription_result = transcription
            st.rerun()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, analyze_audio_clicked
    
    

def render_microphone_preview():
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # Optional: send to backend


def render_results_section(result: Dict[str, Any], transcription: str = None):
    """Render the results section with moderation output matching backend structure."""
    st.markdown("## 📊 Analysis Results")
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # Display transcription if available
    if transcription:
        st.markdown("### 🎙️ Transcription")
        st.info(f"**Transcribed Text**: {transcription}")
    
    # Classification result
    classification = result.get('label', 'Unknown')
    st.markdown("### 🏷️ Classification")
    st.markdown(create_classification_badge(classification), unsafe_allow_html=True)
    
    # Classification reason (from hate speech agent)
    classification_reason = result.get('classification_reason', 'No reason provided')
    st.markdown("### 🤖 Classification Reasoning")
    st.info(classification_reason)
    
    # Detailed explanation (from policy reasoning agent)
    explanation = result.get('explanation', 'No explanation provided')
    st.markdown("### 💡 Policy-Based Explanation")
    st.write(explanation)
    
    # Relevant policy snippets
    policy_snippets = result.get('policy_snippets', [])
    if policy_snippets:
        st.markdown("### 📜 Relevant Policy Snippets")
        for i, policy in enumerate(policy_snippets, 1):
            with st.expander(f"Policy {i}: {policy.get('source_file', 'Unknown Source')} (Score: {policy.get('score', 'N/A')})"):
                st.markdown(f"""
                <div class="policy-snippet">
                <strong>Source:</strong> {policy.get('source_file', 'N/A')}<br>
                <strong>Relevance Score:</strong> {policy.get('score', 'N/A')}<br><br>
                <strong>Content:</strong><br>
                {policy.get('text', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    
    # Recommended action
    action = result.get('action', 'No action specified')
    action_reason = result.get('action_reason', 'No reason provided')
    
    st.markdown("### ⚡ Recommended Moderation Action")
    st.markdown(f"""
    <div class="action-recommendation">
    <strong>Action:</strong> {action}<br>
    <strong>Reason:</strong> {action_reason}
    </div>
    """, unsafe_allow_html=True)
    
    # Export functionality
    export_text = export_results_as_text(result, transcription)
    st.download_button(
        label="📄 Export Results",
        data=export_text,
        file_name=f"moderation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_loading_spinner():
    """Display loading spinner during processing."""
    st.markdown("""
    <div class="loading-spinner">
        <div style="text-align: center;">
            <h3>🔄 Processing...</h3>
            <p>Analyzing content and retrieving relevant policies...</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================
# MAIN APPLICATION LOGIC
# ================================

def process_text_moderation(text: str) -> Optional[Dict[str, Any]]:
    """Process text moderation request matching backend API structure."""
    is_valid, error_msg = validate_text_input(text)
    if not is_valid:
        st.error(error_msg)
        return None
    
    with st.spinner("Analyzing text..."):
        # Backend expects {"text": "..."}
        result = make_api_request(MODERATE_ENDPOINT, {"text": text})
        return result

def process_audio_moderation(audio_file) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Process audio transcription and moderation matching backend API structure."""
    with st.spinner("Transcribing and analyzing audio..."):
        # Backend expects multipart form data with "file" field
        files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
        result = make_api_request(TRANSCRIBE_ENDPOINT, files=files)
        
        if result:
            # Backend returns {"transcribed_text": "...", "moderation_result": {...}}
            transcription = result.get('transcribed_text', '')
            moderation_result = result.get('moderation_result', {})
            return moderation_result, transcription
        
        return None, None

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Hate Speech Detection Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state and load CSS
    initialize_session_state()
    load_css()
    #render_microphone_preview()

    # Render UI components
    render_header()
    # render_sidebar()
    #render_microphone_preview()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input section
        user_input, analyze_text_clicked = render_text_input_section()
        st.divider()
        
        # Audio input section
        uploaded_audio, analyze_audio_clicked = render_audio_section()
        
        # Process text analysis
        if analyze_text_clicked and user_input:
            result = process_text_moderation(user_input)
            if result:
                st.session_state.moderation_result = result
                st.session_state.transcription_result = None
                st.rerun()
        
        # Process audio analysis
        if analyze_audio_clicked and uploaded_audio:
            moderation_result, transcription = process_audio_moderation(uploaded_audio)
            if moderation_result:
                st.session_state.moderation_result = moderation_result
                st.session_state.transcription_result = transcription
                st.rerun()
    
    with col2:
        # Display results if available
        if st.session_state.moderation_result:
            render_results_section(
                st.session_state.moderation_result, 
                st.session_state.transcription_result
            )
        else:
            st.info("👈 Enter text or upload audio to get started with content analysis.")

if __name__ == "__main__":
    main()