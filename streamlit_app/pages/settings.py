import streamlit as st
import os
import shutil
from models.model_registry import list_registered_models
from database.db_manager import get_history_df, clear_all_history
from config.config import RESULTS_DIR, DB_PATH
from utils.logger import get_logger

logger = get_logger("SettingsPage")

def render_settings_page():
    st.markdown("<h2 style='color: #2E7559;'>⚙️ System Settings & Diagnostics</h2>", unsafe_allow_html=True)
    st.write("Diagnose system status, manage local saved files, and clear temp logs.")
    
    st.write("---")
    
    # 1. Model Registry Status
    st.markdown("### 🤖 Deep Learning Models Registry")
    models_status = list_registered_models()
    
    for name, info in models_status.items():
        exists = info["saved_on_disk"]
        status_color = "green" if exists else "orange"
        status_text = "Ready (Saved on Disk)" if exists else "Not Trained (Fresh Instance will compile on load)"
        
        st.markdown(f"**{name}**: <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>", unsafe_allow_html=True)
        st.caption(f"Path: `{info['file_path']}`")
        
    st.write("---")
    
    # 2. Disk & Directory Maintenance
    st.markdown("### 🧹 Disk & Temporary File Cleanup")
    temp_dir = os.path.join(RESULTS_DIR, "temp")
    
    temp_size = 0
    temp_file_count = 0
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                fp = os.path.join(root, f)
                temp_size += os.path.getsize(fp)
                temp_file_count += 1
                
    st.metric("Temporary Files (Reports & Visualizations)", f"{temp_file_count} files", f"{temp_size / (1024*1024):.2f} MB")
    
    if st.button("Clear Temporary Cache", type="secondary"):
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                st.success("Temporary files cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
                logger.error(f"Cache clear error: {e}")
        else:
            st.info("Temporary directory is already empty.")
            
    st.write("---")
    
    # 3. Database Maintenance
    st.markdown("### 🗄️ Database Health")
    db_size = os.path.getsize(DB_PATH) / 1024.0 if os.path.exists(DB_PATH) else 0.0
    
    try:
        df = get_history_df()
        record_count = len(df)
    except Exception:
        record_count = 0
        
    col1, col2 = st.columns(2)
    col1.metric("Database File Size", f"{db_size:.1f} KB")
    col2.metric("Total Logs Stored", f"{record_count} rows")
    
    confirm = st.checkbox("Confirm database reset action", key="db_confirm")
    if st.button("Reset Database Logs", type="primary", disabled=not confirm):
        try:
            clear_all_history()
            st.success("Database logs reset successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting database: {e}")
            logger.error(f"Database reset error: {e}")
