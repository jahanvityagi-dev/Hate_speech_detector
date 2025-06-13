import streamlit as st
def render_stats_cards():
        """Render statistics cards."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">Analyses</div>
            </div>
            """.format(st.session_state.analysis_count), unsafe_allow_html=True)
        
        with col2:
            status = "Active" if st.session_state.moderation_result else "Ready"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">✓</div>
                <div class="stat-label">{status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            last_time = st.session_state.last_analysis_time
            time_str = last_time.strftime("%H:%M") if last_time else "--:--"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{time_str}</div>
                <div class="stat-label">Last Analysis</div>
            </div>
            """, unsafe_allow_html=True)