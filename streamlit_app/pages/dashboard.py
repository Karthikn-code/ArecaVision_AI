import streamlit as st
import os
import json
import pandas as pd
from database.db_manager import get_history_df
from dashboard.analytics import (
    get_kpi_metrics, generate_health_pie_chart,
    generate_disease_frequency_chart, generate_prediction_timeline,
    generate_training_curves, get_model_training_summary
)
from evaluation.evaluator import compare_all_models
from config.config import RESULTS_DIR
from utils.logger import get_logger

logger = get_logger("DashboardPage")

def render_dashboard_page():
    st.markdown("<h2 style='color: #2E7559;'>📈 Analytical & Performance Dashboard</h2>", unsafe_allow_html=True)

    df = get_history_df()

    # --------------------------------------------------
    # TAB 1: Real-time Database Diagnostics
    # TAB 2: Training History Curves
    # TAB 3: Model Performance Benchmarks
    # --------------------------------------------------
    tab_data, tab_training, tab_model = st.tabs([
        "📊 Plantation Analytics",
        "📉 Training History",
        "⚙️ Model Benchmarks & Comparison"
    ])

    with tab_data:
        st.write("Real-time telemetry and statistics compiled from historical diagnosis logs.")

        if df.empty:
            st.info("No prediction data recorded yet. Please run some diagnostics in the 'Disease Detection' page first.")
        else:
            metrics = get_kpi_metrics(df)

            st.write("---")
            # KPI Cards
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Diagnoses", f"{metrics['total_predictions']}")
            col2.metric("Healthy Ratio", f"{metrics['healthy_pct']:.1f}%",
                        delta=f"{metrics['healthy_count']} cases", delta_color="normal")
            col3.metric("Diseased Ratio", f"{metrics['diseased_pct']:.1f}%",
                        delta=f"-{metrics['diseased_count']} cases", delta_color="inverse")
            col4.metric("Avg Speed (ms)", f"{metrics['avg_inference_time_ms']:.1f} ms")
            col5.metric("Top Disease", metrics['most_common_disease'].replace("_", " ").title()
                        if metrics['most_common_disease'] != "None" else "—")

            st.write("---")
            # Plotly Charts Section
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                pie_fig = generate_health_pie_chart(df)
                st.plotly_chart(pie_fig, use_container_width=True)

            with col_chart2:
                bar_fig = generate_disease_frequency_chart(df)
                st.plotly_chart(bar_fig, use_container_width=True)

            st.write("---")
            # Timeline Chart
            timeline_fig = generate_prediction_timeline(df)
            st.plotly_chart(timeline_fig, use_container_width=True)

    # --------------------------------------------------
    # TAB 2: Training History
    # --------------------------------------------------
    with tab_training:
        st.write("Accuracy and loss curves from the two-stage training pipeline for each deep learning model.")

        training_summary = get_model_training_summary(RESULTS_DIR)

        if not training_summary:
            st.info(
                "No training history found. Train models using:\n"
                "```bash\npython training/train.py --all\n```"
            )
        else:
            for model_name, info in training_summary.items():
                with st.expander(
                    f"📊 {model_name} — Best Val Accuracy: {info['best_val_accuracy'] * 100:.2f}% "
                    f"| Epochs: {info['total_epochs']}",
                    expanded=True
                ):
                    fig = generate_training_curves(info["history"], model_name=model_name)
                    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # TAB 3: Model Performance Benchmarks
    # --------------------------------------------------
    with tab_model:
        st.write("Performance evaluation metrics of trained transfer learning networks on the 15% testing split.")

        comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")

        # Load or generate model comparison
        comparison_data = {}
        if os.path.exists(comparison_path):
            try:
                with open(comparison_path, 'r') as f:
                    comparison_data = json.load(f)
            except Exception:
                pass

        if st.button("🔄 Compute / Refresh Model Comparison Benchmarks"):
            with st.spinner("Evaluating all models on testing dataset split..."):
                try:
                    comparison_data = compare_all_models()
                    st.success("Comparison benchmarks updated successfully!")
                except Exception as e:
                    st.error(f"Error compiling benchmarks: {e}")
                    logger.error(f"Benchmark error: {e}")

        if not comparison_data:
            st.info("No benchmark comparison results saved. Click the button above to run evaluations on the testing split.")
        else:
            st.write("---")
            st.markdown("### Model Comparison Table")

            # Format comparison data into DataFrame
            rows = []
            for m_name, m_metrics in comparison_data.items():
                rows.append({
                    "Model": m_name,
                    "Test Accuracy": f"{m_metrics.get('accuracy', 0.0) * 100:.2f}%",
                    "Avg Precision": f"{m_metrics.get('precision', 0.0) * 100:.2f}%",
                    "Avg Recall": f"{m_metrics.get('recall', 0.0) * 100:.2f}%",
                    "Avg F1-Score": f"{m_metrics.get('f1_score', 0.0) * 100:.2f}%",
                    "Inference Speed": f"{m_metrics.get('inference_time_ms', 0.0):.2f} ms/img"
                })
            comp_df = pd.DataFrame(rows)
            st.table(comp_df)

            # Show charts side-by-side
            st.write("---")
            st.markdown("### Metric Visualizations (Confusion Matrices & ROC Curves)")

            col_bench1, col_bench2 = st.columns(2)

            selected_model = col_bench1.selectbox("Select Model to View Graphs", list(comparison_data.keys()))
            clean_name = selected_model.replace('-', '').lower()

            # Confusion matrix
            cm_img_path = os.path.join(RESULTS_DIR, f"{clean_name}_confusion_matrix.png")
            # ROC curve
            roc_img_path = os.path.join(RESULTS_DIR, f"{clean_name}_roc_curve.png")

            with col_bench1:
                if os.path.exists(cm_img_path):
                    st.image(cm_img_path, caption=f"Confusion Matrix - {selected_model}", use_container_width=True)
                else:
                    st.info("No confusion matrix chart generated yet.")

            with col_bench2:
                if os.path.exists(roc_img_path):
                    st.image(roc_img_path, caption=f"ROC Curve - {selected_model}", use_container_width=True)
                else:
                    st.info("No ROC curve chart generated yet.")

            # Per-class ROC AUC table
            roc_auc = comparison_data.get(selected_model, {}).get("roc_auc", {})
            if roc_auc:
                st.write("---")
                st.markdown(f"### Per-Class ROC AUC — {selected_model}")
                auc_rows = [{"Class": cls, "AUC Score": f"{score:.4f}"} for cls, score in roc_auc.items()]
                st.table(pd.DataFrame(auc_rows))
