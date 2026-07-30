import streamlit as st
import os
import uuid
import time
import numpy as np
from PIL import Image
import cv2

from config.config import RESULTS_DIR, DISPLAY_NAMES, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
from models.model_registry import load_or_create_model, ensemble_predict
from database.db_manager import save_prediction
from augmentation.augmentor import preprocess_and_denoise_image
from gradcam.gradcam import compute_gradcam, overlay_heatmap
from recommendation.engine import RecommendationEngine
from recommendation.translations import get_kannada_recommendation
from utils.severity_estimator import estimate_disease_severity
from reports.pdf_generator import generate_pdf_report
from object_detection import LeafDiseaseDetector
from utils.logger import get_logger

logger = get_logger("DetectionPage")

# Initialize object spot detector
detector_engine = LeafDiseaseDetector()


# Confidence threshold below which the system warns of uncertainty
CONFIDENCE_THRESHOLD = 0.50

# Initialize recommendation engine
rec_engine = RecommendationEngine()


def save_uploaded_file(uploaded_file, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    orig_name = getattr(uploaded_file, 'name', 'camera_capture.png')
    if not orig_name:
        orig_name = 'camera_capture.png'
    ext = os.path.splitext(orig_name)[1]
    if not ext:
        ext = ".png"
    filename = f"{file_id}{ext}"
    file_path = os.path.join(dest_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def render_detection_page():
    st.markdown("<h2 style='color: #2E7559;'>🔬 Disease Detection & Diagnostics</h2>", unsafe_allow_html=True)
    st.write("Upload an image or take a live photo of an areca nut leaf, fruit (nut), or trunk to run deep learning diagnostics.")

    # Model Selector
    model_options = [
        "EfficientNet-B0",
        "MobileNetV3",
        "ResNet50",
        "🤝 Ensemble (All Models — Highest Accuracy)"
    ]
    model_name = st.selectbox(
        "Select Model Architecture",
        model_options,
        help="Ensemble mode soft-averages probabilities across all 3 trained models for the most reliable diagnosis."
    )
    use_ensemble = model_name.startswith("🤝 Ensemble")

    # Language Selector
    lang = st.radio("Select Output Language / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ", ["English", "ಕನ್ನಡ (Kannada)"], horizontal=True)
    use_kannada = lang.startswith("ಕನ್ನಡ")

    # Optional Farmer Name Input
    farmer_name = st.text_input("Farmer Name / ರೈತರ ಹೆಸರು (Optional, for PDF Report)", "")

    # Image Input Selector (Upload vs Camera)
    input_source = st.radio("Select Image Input Method", ["Upload File", "Take Photo via Camera"], horizontal=True)

    active_file = None
    if input_source == "Upload File":
        active_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "bmp"])
    else:
        active_file = st.camera_input("Take a photo of the plant part")

    if active_file is not None:
        st.write("---")

        # Temp save paths
        temp_dir = os.path.join(RESULTS_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        orig_path = save_uploaded_file(active_file, temp_dir)

        # 1. Image Validation
        try:
            with Image.open(orig_path) as img:
                img.verify()
            is_valid = True
        except Exception:
            is_valid = False

        if not is_valid:
            st.error("❌ Corrupted or invalid image file. Please upload a clear image in JPG, PNG, or BMP format.")
            if os.path.exists(orig_path):
                os.remove(orig_path)
            return

        # Display processing spinner
        with st.spinner("Processing image and running model diagnostics..."):
            start_time = time.time()

            # 2. Denoising, Resizing — returns float32 in [0, 255]
            try:
                preprocessed_img = preprocess_and_denoise_image(orig_path)
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                return

            # Expand to batch shape (1, 224, 224, 3)
            img_batch = np.expand_dims(preprocessed_img, axis=0)

            # 3. Model Inference
            gradcam_model = None
            try:
                if use_ensemble:
                    predictions, ensemble_models = ensemble_predict(img_batch)
                    # Use EfficientNet-B0 for Grad-CAM (highest individual accuracy)
                    gradcam_model = ensemble_models.get("EfficientNet-B0")
                    used_model_label = "Ensemble (EfficientNet-B0 + MobileNetV3 + ResNet50)"
                else:
                    gradcam_model = load_or_create_model(model_name)
                    predictions = gradcam_model.predict(img_batch, verbose=0)
                    used_model_label = model_name
            except Exception as e:
                st.error(f"Inference error: {e}")
                logger.error(f"Error loading model or running prediction: {e}")
                return

            end_time = time.time()
            processing_time = end_time - start_time

            # Interpret predictions
            pred_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][pred_idx])
            predicted_class = CLASS_NAMES[pred_idx]
            display_name = DISPLAY_NAMES.get(predicted_class, predicted_class)

            # 4. Grad-CAM Heatmap Calculation
            gradcam_path = None
            if gradcam_model is not None:
                try:
                    heatmap = compute_gradcam(gradcam_model, img_batch, class_index=pred_idx)
                    overlay = overlay_heatmap(heatmap, preprocessed_img, alpha=0.5)

                    gradcam_filename = f"gradcam_{uuid.uuid4()}.jpg"
                    gradcam_path = os.path.join(temp_dir, gradcam_filename)

                    # Convert RGB overlay to BGR for OpenCV save
                    overlay_bgr = cv2.cvtColor(np.uint8(255 * overlay), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(gradcam_path, overlay_bgr)
                except Exception as e:
                    logger.error(f"Grad-CAM error: {e}")
                    st.warning("⚠️ Explainable AI heatmap could not be generated for this model layout.")

            # 4b. Object & Spot Detection Pipeline
            detection_res = None
            try:
                detection_res = detector_engine.predict(orig_path)
            except Exception as e:
                logger.error(f"Object detection error: {e}")

            spots_count = detection_res.get("disease_spots_count", 0) if detection_res else 0

            # 5. Save to Database
            pred_id = str(uuid.uuid4())[:8].upper()
            try:
                save_prediction(
                    prediction_id=pred_id,
                    image_path=orig_path,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    processing_time=processing_time,
                    model_used=used_model_label,
                    spots_detected=spots_count
                )
            except Exception as e:
                logger.error(f"Database save error: {e}")

        # 6. UI Results Rendering
        # Confidence threshold warning
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ **Low Confidence ({confidence * 100:.1f}%)** — The model is uncertain about this image. "
                "Please upload a clearer, well-lit image of the areca nut leaf, fruit, or trunk for a reliable diagnosis."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Diagnosis Summary")
            st.markdown(f"**Prediction ID:** `{pred_id}`")

            # Health status badge
            is_healthy = "Healthy" in display_name or "healthy" in predicted_class
            status_color = "#2E7559" if is_healthy else "#B43232"
            status_badge = "✅ HEALTHY" if is_healthy else "⚠️ DISEASE DETECTED"
            st.markdown(
                f"<div style='background: {status_color}22; border: 1.5px solid {status_color}; "
                f"border-radius: 8px; padding: 12px; margin: 8px 0;'>"
                f"<span style='font-size: 18px; color: {status_color}; font-weight: bold;'>"
                f"{status_badge}</span><br/>"
                f"<span style='font-size: 15px;'>{display_name}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Disease Severity Estimation
            raw_heatmap = heatmap if 'heatmap' in locals() else None
            severity_info = estimate_disease_severity(preprocessed_img, predicted_class, heatmap_2d=raw_heatmap)

            st.markdown(
                f"<div style='background: {severity_info['status_color']}22; border: 1.5px solid {severity_info['status_color']}; "
                f"border-radius: 8px; padding: 10px; margin: 8px 0;'>"
                f"<strong style='color: {severity_info['status_color']};'>Disease Severity: {severity_info['severity_level']} "
                f"({severity_info['severity_pct']}%)</strong><br/>"
                f"<span style='font-size: 13px; color: #BBB;'>{severity_info['description']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            st.progress(confidence)
            st.markdown(f"**Diseased Spots Located:** `{spots_count}`")
            st.markdown(f"**Inference Speed:** `{processing_time * 1000:.1f} ms`")
            st.markdown(f"**Active Model:** `{used_model_label}`")

            # Per-class probability expander
            with st.expander("📊 Full Class Probability Distribution"):
                for i, cls in enumerate(CLASS_NAMES):
                    disp = DISPLAY_NAMES.get(cls, cls)
                    prob = float(predictions[0][i])
                    st.markdown(f"**{disp}**: `{prob * 100:.2f}%`")
                    st.progress(prob)

        with col2:
            st.markdown("### Visual Explanation")
            tab1, tab2, tab3, tab4 = st.tabs(["Raw Upload", "Bilateral Denoised", "Grad-CAM Overlay", "🎯 Object & Spot Bounding Boxes"])
            with tab1:
                st.image(orig_path, use_container_width=True, caption="Original Raw Image Uploaded")
            with tab2:
                disp_img = preprocessed_img / 255.0 if preprocessed_img.max() > 1.0 else preprocessed_img
                st.image(disp_img, use_container_width=True, caption="OpenCV Bilateral Denoised (224x224)")
                st.info("Bilateral filtering reduces pixel noise/artifacts while keeping edge borders sharp.")
            with tab3:
                if gradcam_path and os.path.exists(gradcam_path):
                    st.image(gradcam_path, use_container_width=True,
                             caption="Attention map highlighting regions influencing prediction")
                else:
                    st.info("No Grad-CAM overlay available.")
            with tab4:
                if detection_res and "annotated_image" in detection_res:
                    st.image(detection_res["annotated_image"], use_container_width=True, caption="Color-Coded Bounding Boxes & Spot Localization")
                    st.success(f"🎯 Total Objects/Spots Detected: {detection_res.get('total_boxes_count', 0)}")
                    if detection_res.get("boxes"):
                        with st.expander("🔍 Detailed Bounding Box Coordinates"):
                            st.json(detection_res["boxes"])
                else:
                    st.info("Object detection output unavailable.")

        # 7. Recommendations Section
        st.write("---")
        if use_kannada:
            st.markdown("### 📋 ಕೃಷಿ ಪದ್ಧತಿ ಮತ್ತು ಚಿಕಿತ್ಸಾ ಕೈಪಿಡಿ (Kannada Guide)")
            rec_details = get_kannada_recommendation(predicted_class)
        else:
            st.markdown("### 📋 Agronomic Treatment & Prevention Sheets")
            rec_details = rec_engine.get_recommendation(predicted_class)

        st.markdown(f"**Scientific Name:** *{rec_details.get('scientific_name', 'N/A')}*")
        st.markdown(f"**Description:** {rec_details.get('description', '')}")
        st.markdown(f"**Probable Cause:** {rec_details.get('cause', 'N/A')}")

        st.markdown("**Key Symptoms:**")
        for sym in rec_details.get("symptoms", []):
            st.markdown(f"- {sym}")

        # Tabs for Controls
        tab_org, tab_chem, tab_prev = st.tabs(["Organic Control", "Chemical Control", "Preventive Measures"])

        with tab_org:
            st.markdown("**Organic / Biological Management:**")
            st.write(rec_details.get("treatment", {}).get("organic_control", "No organic treatments listed."))

        with tab_chem:
            st.markdown("**Chemical Management:**")
            st.write(rec_details.get("treatment", {}).get("chemical_control", "No chemical treatments listed."))
            st.markdown("---")
            col_f, col_p = st.columns(2)
            col_f.metric("Recommended Fungicide",
                         rec_details.get("treatment", {}).get("recommended_fungicide", "N/A"))
            col_p.metric("Recommended Pesticide",
                         rec_details.get("treatment", {}).get("recommended_pesticide", "N/A"))

        with tab_prev:
            st.markdown("**Long-term Prevention:**")
            for idx, prev in enumerate(rec_details.get("preventive_measures", [])):
                st.markdown(f"{idx+1}. {prev}")

        # 8. Report Generation Exporter
        st.write("---")
        st.markdown("### 📄 Generate Professional Diagnostic Report")

        pdf_filename = f"report_{pred_id}.pdf"
        pdf_dest_path = os.path.join(temp_dir, pdf_filename)

        try:
            generate_pdf_report(
                dest_path=pdf_dest_path,
                farmer_name=farmer_name,
                original_img_path=orig_path,
                gradcam_img_path=gradcam_path if gradcam_path else "",
                predicted_disease=predicted_class,
                confidence=confidence,
                processing_time=processing_time,
                model_used=used_model_label,
                rec_details=rec_details
            )

            with open(pdf_dest_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            st.download_button(
                label="📥 Download Diagnostic PDF Report",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Could not construct PDF report: {e}")
            logger.error(f"PDF compilation error: {e}")
