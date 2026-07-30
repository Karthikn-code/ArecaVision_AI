import streamlit as st
import os
import json
from config.config import RESULTS_DIR

def render_documentation_page():
    st.markdown("<h2 style='color: #2E7559;'>📖 System Documentation & Manual</h2>", unsafe_allow_html=True)
    st.write("Complete system guide, architecture description, and setup operations.")
    
    st.write("---")
    
    # Navigation sub-tabs
    tab_guide, tab_arch, tab_train, tab_api = st.tabs([
        "🚀 Quickstart Manual",
        "📐 System Architecture",
        "🏋️ Training Guide",
        "🔑 Model Accuracy"
    ])
    
    with tab_guide:
        st.markdown("""
        ### How to Run a Diagnosis:
        1. Click on the **Disease Detection** tab in the sidebar.
        2. Enter the **Farmer Name** (optional) to stamp the generated report.
        3. Select your desired deep learning architecture:
           - **EfficientNet-B0** — fastest inference, high accuracy
           - **MobileNetV3** — lightweight and efficient
           - **ResNet50** — deep residual network
           - **🤝 Ensemble (All Models)** — soft-votes across all 3 for the highest accuracy
        4. Upload an image of an areca nut leaf, fruit, or trunk (drag-and-drop or select file).
        5. Review the **Predicted Class**, **Health Status Badge**, and **Confidence Score** display.
        6. Check the **Full Class Probability Distribution** expander for per-class breakdown.
        7. ⚠️ If confidence is below **50%**, the system will warn you to upload a clearer image.
        8. Slide over the **Visual Explanation** tab to inspect the **Grad-CAM Overlay Heatmap**.
        9. Review recommendations for Symptoms, Organic Treatment, Chemical Control, and Preventive Measures.
        10. Click **Download Diagnostic PDF Report** to export a formatted diagnostic sheet.
        """)
        
    with tab_arch:
        st.markdown("### System Component Layout")
        st.markdown("""
        The system follows a strict **Modular Layered Architecture**:
        * **Presentation Layer**: Streamlit web frontend displaying responsive telemetry, Plotly analytics, and file inputs.
        * **Business Logic Layer**:
            * `data_prep.py`: Sanitizes and splits original files.
            * `augmentor.py`: Applies Bilateral filter denoising. Images are kept in [0, 255] float32 — no external rescaling.
            * `gradcam.py`: Dynamically extracts weights and feature maps to compute attention overlays.
            * `model_registry.py`: Manages loading, creating, and **Ensemble prediction** across all models.
        * **Data Access Layer**: `db_manager.py` handling SQLite logging.
        * **Storage Layer**: Directory-based datasets, saved models (`.keras` files), JSON-backed recommendation engine, and SQLite databases.
        """)
        
        # ASCII architecture diagram
        st.code(r"""
        [ Farmer Image Upload ] ---> [ Streamlit Frontend ]
                                            |
                                            v
                                 [ Preprocessor & Denoise ] (OpenCV Bilateral, [0,255] float32)
                                            |
                                            v
                             +------------------------------+
                             |  Model Selection             |
                             | EfficientNet-B0 / MobileNetV3|
                             | ResNet50 / Ensemble           |
                             +------------------------------+
                                 |                    |
                                 v                    v
                      [ Predicted Class ]      [ Grad-CAM Map ]
                               |                     |
                               v                     v
                    [ Recommendation JSON ]   [ Overlay Image ]
                               \                     /
                                v                   v
                            [ SQLite History ] [ PDF Generator ]
        """)
        
    with tab_train:
        st.markdown("""
        ### How to Train the Deep Learning Models:
        The repository includes a full two-stage training pipeline with **class-weighted loss** and 
        **backbone fine-tuning** for maximum accuracy.
        
        #### Prerequisite:
        Ensure the original dataset is placed at:
        ```
        d:\\Areca\\archive\\Arecanut_dataset\\Arecanut_dataset
        ```
        
        #### Running Training:
        Execute the training wrapper script from your terminal:
        ```bash
        # Activate virtual environment
        .venv\\Scripts\\activate
        
        # Train all 3 models sequentially (recommended — highest accuracy)
        python training/train.py --all --epochs 10 --batch_size 32
        
        # Train a specific model only
        python training/train.py --model EfficientNet-B0 --epochs 10
        python training/train.py --model MobileNetV3 --epochs 10
        python training/train.py --model ResNet50 --epochs 10
        
        # Custom hyperparameters
        python training/train.py --model ResNet50 --epochs 20 --batch_size 16 --lr 5e-5
        ```
        
        #### What the Training Pipeline Does:
        The two-stage training automatically:
        1. **Stage 1 — Head Warmup** (5 epochs): Freezes the backbone; trains only the new classification head 
           with class-weighted cross-entropy loss to handle dataset imbalance.
        2. **Stage 2 — Backbone Fine-Tuning** (10 epochs): Unfreezes the top 30 layers of the backbone 
           and fine-tunes end-to-end with a reduced learning rate (1e-5).
        3. Verifies and splits the dataset 70/15/15 into `results/split_dataset/`.
        4. Saves best checkpoints based on validation accuracy.
        5. Outputs training history JSON files to `results/` for dashboard graphing.
        
        #### Expected Accuracy:
        | Model | Val Accuracy (target) |
        |---|---|
        | EfficientNet-B0 | ~98-99% |
        | MobileNetV3 | ~95-97% |
        | ResNet50 | ~94-97% |
        | **Ensemble** | **~98-99%** |
        """)
        
    with tab_api:
        st.markdown("### Model Accuracy Metrics")
        
        # Try to load model comparison results
        comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")
        
        if os.path.exists(comparison_path):
            try:
                with open(comparison_path, "r") as f:
                    comparison = json.load(f)
                    
                for model_name, metrics in comparison.items():
                    with st.expander(f"📊 {model_name}", expanded=True):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Test Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
                        c2.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
                        c3.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")
                        c4.metric("F1 Score", f"{metrics.get('f1_score', 0) * 100:.2f}%")
            except Exception as e:
                st.warning(f"Could not load model comparison file: {e}")
        else:
            st.info(
                "📋 Model accuracy metrics will appear here after running the evaluator:\n\n"
                "```bash\n"
                "python evaluation/evaluator.py\n"
                "```\n\n"
                "This generates `results/model_comparison.json` with test-set accuracy, precision, recall, and F1 scores."
            )
    
    st.write("---")
    st.info("💡 Tip: Full implementation details are documented in the project's root `README.md` and `documentation/system_architecture.md` files.")
