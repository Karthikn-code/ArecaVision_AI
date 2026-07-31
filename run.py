"""
run.py — Project entry point for ArecaVision AI.

Launches the Streamlit application from the project root directory.
Ensures sys.path is correctly configured for all module imports.

Usage:
    python run.py
    # or directly via streamlit:
    streamlit run run.py
"""

import os
import sys

# Guarantee the project root is the first entry on sys.path
# so all relative package imports (config, models, database, etc.) resolve correctly.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.db_manager import init_db
from streamlit_app.pages.home import render_home_page
from streamlit_app.pages.detection import render_detection_page
from streamlit_app.pages.history import render_history_page
from streamlit_app.pages.dashboard import render_dashboard_page
from streamlit_app.pages.about import render_about_page
from streamlit_app.pages.documentation import render_documentation_page
from streamlit_app.pages.settings import render_settings_page
from streamlit_app.components.chatbot_widget import render_chatbot_widget

import streamlit as st

# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ArecaVision AI - Health Monitoring & Diagnosis",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Global Premium Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main body background */
    .stApp {
        background-color: #0F1311;
        color: #E0E0E0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161D19 !important;
        border-right: 1px solid #2E7559;
    }

    /* Heading typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }

    /* Button hover effects */
    div.stButton > button {
        background-color: #2E7559;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #3C9370;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(46, 117, 89, 0.4);
    }

    /* Notification styling */
    div[data-testid="stNotification"] {
        background-color: #1A2420 !important;
        border: 1px solid #2E7559 !important;
        border-radius: 8px !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] {
        background-color: #1C2421 !important;
        border-radius: 8px !important;
    }

    /* Tables */
    table {
        background-color: #161D19 !important;
        color: #E0E0E0 !important;
        border-collapse: collapse !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    th {
        background-color: #2E7559 !important;
        color: white !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────
def main():
    # Initialize SQLite database on first run
    init_db()

    # Sidebar branding
    st.sidebar.markdown(
        "<h2 style='text-align: center; color: #2E7559;'>🌴 ArecaVision AI</h2>",
        unsafe_allow_html=True
    )
    st.sidebar.write("---")

    # Navigation
    pages = {
        "🏠 Home": render_home_page,
        "🔬 Disease Detection": render_detection_page,
        "📜 Prediction History": render_history_page,
        "📈 Dashboard": render_dashboard_page,
        "💬 ArecaBot Chat Assistant": lambda: render_chatbot_page(),
        "ℹ️ About": render_about_page,
        "📖 Documentation": render_documentation_page,
        "⚙️ Settings": render_settings_page
    }

    selection = st.sidebar.radio("Navigation Menu", list(pages.keys()))

    st.sidebar.write("---")
    
    # Render persistent Floating Chatbot Widget in Sidebar
    st.sidebar.markdown("### 🤖 ArecaBot Assistant")
    with st.sidebar:
        render_chatbot_widget()

    st.sidebar.write("---")
    st.sidebar.caption("© 2026 ArecaVision AI System")
    st.sidebar.caption("Final Year AI & Data Science Project")

    pages[selection]()


def render_chatbot_page():
    st.markdown("<h2 style='color: #2E7559;'>💬 ArecaBot — AI Agronomist Assistant</h2>", unsafe_allow_html=True)
    st.write("Ask any questions regarding Areca palm nut, leaf, trunk, and root health, organic remedies, fungicide dosages, or fertilizer schedules.")
    st.write("---")
    render_chatbot_widget()


if __name__ == "__main__":
    main()
