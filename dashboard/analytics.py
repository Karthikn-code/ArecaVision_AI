import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.logger import get_logger

logger = get_logger("AnalyticsDashboard")

# Define categories
HEALTHY_CLASSES = ["healthy_foot", "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk"]

def get_kpi_metrics(df):
    """
    Computes key performance metrics from the historical prediction logs.
    """
    if df.empty:
        return {
            "total_predictions": 0,
            "healthy_count": 0,
            "healthy_pct": 0.0,
            "diseased_count": 0,
            "diseased_pct": 0.0,
            "most_common_disease": "None",
            "avg_inference_time_ms": 0.0
        }
        
    total = len(df)
    
    # Categorize healthy vs diseased
    df['is_healthy'] = df['predicted_class'].isin(HEALTHY_CLASSES)
    healthy_count = df['is_healthy'].sum()
    diseased_count = total - healthy_count
    
    healthy_pct = (healthy_count / total) * 100.0
    diseased_pct = (diseased_count / total) * 100.0
    
    # Find most common disease (excluding healthy categories)
    diseased_df = df[~df['predicted_class'].isin(HEALTHY_CLASSES)]
    if not diseased_df.empty:
        most_common_disease = diseased_df['predicted_class'].mode().iloc[0]
    else:
        most_common_disease = "None"
        
    avg_inference = df['processing_time'].mean() * 1000.0 # convert to ms
    
    return {
        "total_predictions": total,
        "healthy_count": int(healthy_count),
        "healthy_pct": float(healthy_pct),
        "diseased_count": int(diseased_count),
        "diseased_pct": float(diseased_pct),
        "most_common_disease": most_common_disease,
        "avg_inference_time_ms": float(avg_inference)
    }

def generate_health_pie_chart(df):
    """
    Generates a Plotly Pie Chart representing the proportion of healthy vs diseased arecanut samples.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
        
    df = df.copy()
    df['Status'] = df['predicted_class'].apply(lambda x: 'Healthy' if x in HEALTHY_CLASSES else 'Diseased')
    counts = df['Status'].value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    
    fig = px.pie(
        counts, 
        names='Status', 
        values='Count',
        color='Status',
        color_discrete_map={'Healthy': '#2E7559', 'Diseased': '#B43232'},  # Custom Forest Green and Red
        hole=0.4,
        title="Overall Plantation Health Status"
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#1E1E1E', width=1)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
    return fig

def generate_disease_frequency_chart(df):
    """
    Generates a Plotly Bar Chart representing the frequency of detected conditions.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
        
    counts = df['predicted_class'].value_counts().reset_index()
    counts.columns = ['Condition', 'Detections']
    
    # Sort for visual aesthetics
    counts = counts.sort_values(by='Detections', ascending=True)
    
    fig = px.bar(
        counts,
        y='Condition',
        x='Detections',
        orientation='h',
        color='Condition',
        color_discrete_sequence=px.colors.qualitative.Pastel2,
        title="Distribution of Diagnosed Conditions"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)')
    )
    return fig

def generate_prediction_timeline(df):
    """
    Generates a Plotly Line Chart representing the prediction timeline.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
        
    df = df.copy()
    # Group by date
    timeline = df.groupby('date').size().reset_index(name='Predictions')
    timeline = timeline.sort_values(by='date')
    
    fig = px.line(
        timeline,
        x='date',
        y='Predictions',
        markers=True,
        title="Diagnostic Activity Timeline"
    )
    fig.update_traces(line_color='#2E7559', marker=dict(size=8, color='#E0E0E0'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='#333333', title="Date"),
        yaxis=dict(gridcolor='#333333', title="Daily Diagnostic Actions")
    )
    return fig


def generate_training_curves(history: dict, model_name: str = "Model"):
    """
    Generates dual Plotly subplots for accuracy and loss over training epochs
    from a training history dict (loaded from *_history.json).

    Args:
        history: Dict with keys like 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        model_name: Label shown in the chart title.

    Returns:
        fig: Plotly Figure with two subplots (accuracy + loss).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    epochs = list(range(1, len(history.get("accuracy", history.get("val_accuracy", []))) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{model_name} — Accuracy", f"{model_name} — Loss")
    )

    # --- Accuracy subplot ---
    if "accuracy" in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history["accuracy"], mode="lines+markers",
                       name="Train Accuracy", line=dict(color="#2E7559", width=2),
                       marker=dict(size=5)),
            row=1, col=1
        )
    if "val_accuracy" in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history["val_accuracy"], mode="lines+markers",
                       name="Val Accuracy", line=dict(color="#5EC4A0", width=2, dash="dash"),
                       marker=dict(size=5)),
            row=1, col=1
        )

    # --- Loss subplot ---
    if "loss" in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history["loss"], mode="lines+markers",
                       name="Train Loss", line=dict(color="#B43232", width=2),
                       marker=dict(size=5)),
            row=1, col=2
        )
    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers",
                       name="Val Loss", line=dict(color="#E07070", width=2, dash="dash"),
                       marker=dict(size=5)),
            row=1, col=2
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#333333', borderwidth=1),
        height=350
    )
    fig.update_xaxes(gridcolor='#333333', title_text="Epoch")
    fig.update_yaxes(gridcolor='#333333')

    return fig


def get_model_training_summary(results_dir: str) -> dict:
    """
    Scans the results/ directory for *_history.json files and returns a
    summary dict: {model_name: {"best_val_accuracy": float, "total_epochs": int}}.

    Args:
        results_dir: Absolute path to the results directory.

    Returns:
        summary dict keyed by clean model name.
    """
    import json
    import os

    model_names = {
        "efficientnetb0": "EfficientNet-B0",
        "mobilenetv3": "MobileNetV3",
        "resnet50": "ResNet50"
    }
    summary = {}

    for clean_name, display_name in model_names.items():
        hist_path = os.path.join(results_dir, f"{clean_name}_history.json")
        if not os.path.exists(hist_path):
            continue
        try:
            with open(hist_path, "r") as f:
                history = json.load(f)
            val_accs = history.get("val_accuracy", [])
            summary[display_name] = {
                "best_val_accuracy": float(max(val_accs)) if val_accs else 0.0,
                "total_epochs": len(val_accs),
                "history": history
            }
        except Exception as e:
            logger.error(f"Failed to load training history for {display_name}: {e}")

    return summary
