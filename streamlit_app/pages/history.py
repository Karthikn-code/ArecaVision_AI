import streamlit as st
import pandas as pd
from database.db_manager import get_history_df, delete_prediction, clear_all_history
from utils.logger import get_logger

logger = get_logger("HistoryPage")

ROWS_PER_PAGE = 20  # Number of records per page for pagination


def render_history_page():
    st.markdown("<h2 style='color: #2E7559;'>📜 Prediction History Log</h2>", unsafe_allow_html=True)
    st.write("View, search, filter, and manage previous diagnosis records saved in the local SQLite database.")

    df = get_history_df()

    if df.empty:
        st.info("No prediction history found. Go to the 'Disease Detection' page to diagnose some images.")
        return

    st.write("---")

    # ─── Filtering tools ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    classes_list = ["All"] + list(df['predicted_class'].unique())
    selected_class = col1.selectbox("Filter by Condition", classes_list)

    models_list = ["All"] + list(df['model_used'].unique())
    selected_model = col2.selectbox("Filter by Model", models_list)

    search_query = col3.text_input("Search by Prediction ID", "").strip().upper()

    # ─── Apply filters ─────────────────────────────────────────────────────
    filtered_df = df.copy()
    if selected_class != "All":
        filtered_df = filtered_df[filtered_df['predicted_class'] == selected_class]
    if selected_model != "All":
        filtered_df = filtered_df[filtered_df['model_used'] == selected_model]
    if search_query:
        filtered_df = filtered_df[filtered_df['prediction_id'].str.contains(search_query, na=False)]

    total_records = len(filtered_df)
    st.write(f"Showing **{total_records}** records matching the filters:")

    # ─── Rename columns for display ────────────────────────────────────────
    display_df = filtered_df.copy()
    display_df.rename(columns={
        "prediction_id": "Prediction ID",
        "predicted_class": "Diagnosed Condition",
        "confidence": "Confidence Score",
        "date": "Date",
        "time": "Time",
        "processing_time": "Processing Time (s)",
        "model_used": "Model Used",
        "image_path": "Local File Path"
    }, inplace=True)

    # ─── Pagination ────────────────────────────────────────────────────────
    total_pages = max(1, (total_records + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)

    if total_records > ROWS_PER_PAGE:
        page_num = st.number_input(
            f"Page (1–{total_pages})",
            min_value=1, max_value=total_pages, value=1, step=1
        )
    else:
        page_num = 1

    start_row = (page_num - 1) * ROWS_PER_PAGE
    end_row = start_row + ROWS_PER_PAGE
    paged_df = display_df.iloc[start_row:end_row]

    st.dataframe(paged_df, use_container_width=True)

    if total_pages > 1:
        st.caption(f"Page {page_num} of {total_pages} | Rows {start_row + 1}–{min(end_row, total_records)} of {total_records}")

    # ─── CSV Export ────────────────────────────────────────────────────────
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export All Logs to CSV",
        data=csv_data,
        file_name="arecavision_diagnosis_logs.csv",
        mime="text/csv"
    )

    # ─── Quick Statistics ──────────────────────────────────────────────────
    if total_records > 0:
        st.write("---")
        st.markdown("### 📊 Quick Statistics")
        qcol1, qcol2, qcol3 = st.columns(3)

        avg_conf = filtered_df['confidence'].mean() * 100
        avg_time_ms = filtered_df['processing_time'].mean() * 1000
        most_common = filtered_df['predicted_class'].mode().iloc[0] if not filtered_df.empty else "—"

        qcol1.metric("Avg Confidence", f"{avg_conf:.1f}%")
        qcol2.metric("Avg Processing Time", f"{avg_time_ms:.1f} ms")
        qcol3.metric("Most Common Condition", most_common.replace("_", " ").title())

    # ─── Management / Deletion ─────────────────────────────────────────────
    st.write("---")
    st.markdown("### ⚙️ Log Database Management")

    del_col, clear_col = st.columns(2)

    with del_col:
        st.markdown("#### Delete Individual Record")
        to_delete = st.text_input("Enter Prediction ID to delete", "").strip().upper()
        if st.button("Delete Record", type="secondary"):
            if to_delete:
                if to_delete in df['prediction_id'].values:
                    try:
                        delete_prediction(to_delete)
                        st.success(f"Record `{to_delete}` deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting record: {e}")
                else:
                    st.warning(f"Prediction ID `{to_delete}` not found in the database.")
            else:
                st.warning("Please enter a valid Prediction ID.")

    with clear_col:
        st.markdown("#### Danger Zone")
        st.write("Permanently delete all historical logs from the database.")
        confirm = st.checkbox("I confirm that I want to delete the entire database history.")
        if st.button("Clear All History", type="primary", disabled=not confirm):
            try:
                clear_all_history()
                st.success("All historical logs deleted successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing history: {e}")
