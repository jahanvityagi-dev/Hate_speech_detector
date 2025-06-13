import streamlit as st

def load_custom_styles():
    """Load and apply custom CSS styles."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def get_custom_css() -> str:
    """Return custom CSS for the application."""
    return """
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #4f46e5;
        --secondary-color: #7c3aed;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --neutral-color: #6b7280;
        --background-light: #f8fafc;
        ---background-card: #1e1e1e;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border-color: #2d2d2d;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styles */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.025em;
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.25rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Card components */
    .card {
        background: var(--background-card);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid var(--border-color);
        box-shadow: none;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    /* Classification badges */
    .classification-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.5rem;
        border-radius: 9999px;
        color: white;
        font-weight: 600;
        font-size: 1.125rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease;
    }
    
    .classification-badge:hover {
        transform: scale(1.05);
    }
    
    /* Results container */
    .results-container {
        background: var(--background-card);
        color: var(--text-primary);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
    }
    
    /* Policy snippets */
    .policy-snippet {
        background: var(--background-card);
        border: 2px solid var(--success-color);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--success-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .policy-snippet:hover {
        box-shadow: var(--shadow-md);
        transform: translateX(4px);
    }
    
    .policy-snippet strong {
        color: var(--success-color);
        font-weight: 600;
    }
    
    /* Action recommendation */
    .action-recommendation {
        background: var(--background-card);
        border: 2px solid var(--warning-color);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--warning-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .action-recommendation:hover {
        box-shadow: var(--shadow-md);
        transform: translateX(4px);
    }
    
    .action-recommendation strong {
        color: var(--warning-color);
        font-weight: 600;
    }
    
    /* Audio section */
    .audio-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #bae6fd;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Stats and info boxes */
    .info-box {
        background: var(--background-card);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        box-shadow: var(--shadow-sm);
    }
    
    .stat-card {
        background: var(--background-card);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Loading states */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        text-align: center;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: var(--radius-md);
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Text area enhancements */
    .stTextArea textarea {
        border-radius: var(--radius-md);
        border: 2px solid var(--border-color);
        transition: border-color 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgb(79 70 229 / 0.1);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .results-container {
            padding: 1.5rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
    """
