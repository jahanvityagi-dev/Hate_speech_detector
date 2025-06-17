import streamlit as st
import requests
from frontend.config import Config
from frontend.core.enums import ClassificationLabel


def render_sidebar_info():
    """Render sidebar with app information, labels, and backend health."""
    with st.sidebar:
        st.markdown("## 📋 About This Tool")
        st.markdown("""
        <div class="info-box">
            This system uses advanced AI to detect and classify potentially harmful content,
            providing policy-based explanations and moderation recommendations.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🏷️ Classification Types")
        for label_enum in ClassificationLabel:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div style="width: 20px; height: 20px; background-color: {label_enum.color};
                           border-radius: 50%; margin-right: 0.5rem;"></div>
                <strong>{label_enum.label}</strong>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("## ⚙️ System Status")
        try:
            response = requests.get(f"{Config.BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("🟢 Backend Connected")
            else:
                st.warning("🟡 Backend Issues")
        except Exception:
            st.error("🔴 Backend Offline")

        st.markdown("## ⚠️ Disclaimer")
        st.markdown("""
        <div class="info-box">
            This tool is for educational and demonstration purposes. 
            Always review AI decisions critically and consider human oversight 
            for production moderation systems.
        </div>
        """, unsafe_allow_html=True)
