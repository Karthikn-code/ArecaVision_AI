import streamlit as st
import os
import json
from config.config import DISPLAY_NAMES, CLASS_NAMES, RESULTS_DIR
from models.model_registry import list_registered_models
from database.db_manager import get_history_df

def _load_best_val_accuracy(model_name):
    """Load best val_accuracy from training history JSON for a model."""
    clean = model_name.replace("-", "").replace(" ", "").lower()
    hist_path = os.path.join(RESULTS_DIR, f"{clean}_history.json")
    if not os.path.exists(hist_path):
        return None
    try:
        with open(hist_path, "r") as f:
            history = json.load(f)
        val_accs = history.get("val_accuracy", [])
        if val_accs:
            return max(val_accs)
    except Exception:
        pass
    return None

def render_home_page():
    st.markdown("<h1 style='text-align: center; color: #2E7559;'>🌴 ArecaVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #888;'>AI-Powered Areca Nut Health Monitoring and Disease Diagnosis System</h3>", unsafe_allow_html=True)
    st.write("---")

    # Welcome Banner
    st.markdown("""
    ### Welcome to ArecaVision AI
    This system is a state-of-the-art agricultural decision-support application designed for farmers, plantation owners, agricultural researchers, and students.
    By utilizing advanced **Deep Learning (CNNs)** and **Computer Vision**, ArecaVision AI automatically identifies leaf, nut, and trunk diseases in areca nut palms and generates actionable recommendations and explainable AI insights (Grad-CAM).
    """)

    # Metric Columns for System Status
    st.write("### 📊 System Status Dashboard")
    col1, col2, col3 = st.columns(3)
    
    # 1. Models registered
    models_status = list_registered_models()
    trained_count = sum(1 for status in models_status.values() if status["saved_on_disk"])
    col1.metric("Registered Models", f"{trained_count} / {len(models_status)}", help="Models compiled and saved in the workspace.")

    # 2. Database count
    try:
        df = get_history_df()
        history_count = len(df)
    except Exception:
        history_count = 0
    col2.metric("Total Diagnoses Run", f"{history_count} logs", help="Total records saved in SQLite history.")

    # 3. Class categories count
    col3.metric("Supported Categories", f"{len(CLASS_NAMES)} classes", help="Number of health and disease classes mapped in the system.")

    # Live Training Accuracy Section
    st.write("---")
    st.write("### 🎯 Model Training Accuracy")
    
    MODEL_NAMES = ["EfficientNet-B0", "MobileNetV3", "ResNet50"]
    acc_cols = st.columns(len(MODEL_NAMES))
    
    any_history_found = False
    for i, mname in enumerate(MODEL_NAMES):
        best_acc = _load_best_val_accuracy(mname)
        saved = models_status.get(mname, {}).get("saved_on_disk", False)
        
        if best_acc is not None:
            any_history_found = True
            delta_str = "✅ Trained" if saved else "⚙️ Training..."
            acc_cols[i].metric(
                label=mname,
                value=f"{best_acc * 100:.2f}%",
                delta=delta_str,
                help=f"Best validation accuracy achieved during training for {mname}."
            )
        elif saved:
            acc_cols[i].metric(
                label=mname,
                value="Model Saved",
                delta="📦 No history",
                help=f"Model file exists but training history JSON was not found."
            )
        else:
            acc_cols[i].metric(
                label=mname,
                value="Not Trained",
                delta="⏳ Pending",
                help=f"Model has not been trained yet."
            )
    
    if not any_history_found:
        st.info("💡 Run `python training/train.py --all` to train all models and populate accuracy metrics here.")

    # Workflow Section
    st.write("---")
    st.write("### 🔄 End-to-End Diagnostic Workflow")
    st.markdown("""
    1. **Image Upload**: Upload a leaf, nut, or trunk image (JPG, PNG, BMP).
    2. **Validation**: The system validates headers to verify the image is not corrupted.
    3. **Preprocessing**: The image is denoised using a Bilateral Filter and resized to 224×224.
    4. **Model Inference**: Deep neural networks compute class probabilities and confidence scores.
       - If confidence is below **50%**, the system warns about uncertain predictions.
    5. **Explainability**: **Grad-CAM** generates an attention heatmap to show which physical regions influenced the prediction.
    6. **Decision Support**: Retrieval of scientific details, symptoms, and organic/chemical control measures.
    7. **Report Generation**: Export a professional PDF report containing the original image, Grad-CAM heatmap, and custom treatment sheets.
    """)

    # Disease Overview Factsheet
    st.write("---")
    st.markdown("### 📋 Disease & Health Categories (14 Classes)")
    st.caption("🔒 *Important: Class indices follow TensorFlow's case-sensitive alphabetical sort of folder names.*")

    st.markdown("""
| Index | Class Name | Display Name | Category |
| :---: | :--- | :--- | :--- |
| **0** | `Arecanut_YellowBrownSpot` | Yellow Brown Leaf Spot | ⚠️ **Fungal Disease** |
| **1** | `CCI_Caterpillars` | Caterpillar Foliage Infestation | ⚠️ **Pest Infestation** |
| **2** | `Healthy_Leaf` | Healthy Leaf | ✅ **Healthy** |
| **3** | `Healthy_Nut` | Healthy Nut | ✅ **Healthy** |
| **4** | `Healthy_Trunk` | Healthy Trunk | ✅ **Healthy** |
| **5** | `Mahali_Koleroga` | Mahali / Koleroga (Fruit Rot) | ⚠️ **Fungal Disease** |
| **6** | `Stem_bleeding` | Stem Bleeding | ⚠️ **Fungal Disease** |
| **7** | `WCLWD_DryingofLeaflets` | Leaf Wilt / WCLWD (Drying) | ⚠️ **Phytoplasmal** |
| **8** | `WCLWD_Flaccidity` | Leaf Wilt / WCLWD (Drooping) | ⚠️ **Phytoplasmal** |
| **9** | `WCLWD_Yellowing` | Leaf Wilt / WCLWD (Yellowing) | ⚠️ **Phytoplasmal** |
| **10** | `bud borer` | Bud Borer (Pest) | ⚠️ **Pest Infestation** |
| **11** | `healthy_foot` | Healthy Foot / Base | ✅ **Healthy** |
| **12** | `stem cracking` | Stem Cracking | ⚠️ **Structural Disorder** |
| **13** | `yellow leaf disease` | Yellow Leaf Disease | ⚠️ **Phytoplasmal** |
""", unsafe_allow_html=True)

