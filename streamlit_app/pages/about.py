import streamlit as st

def render_about_page():
    st.markdown("<h2 style='color: #2E7559;'>ℹ️ About the Project</h2>", unsafe_allow_html=True)
    st.write("Academic profile and engineering details of the ArecaVision AI system.")
    
    st.write("---")
    
    # Project Info Section
    st.markdown("""
    ### 🌴 Project Overview
    **Title:** AI-Powered Areca Nut Health Monitoring and Disease Diagnosis System Using Deep Learning and Computer Vision
    
    **Scope:** This application serves as a final-year engineering project for **Artificial Intelligence & Data Science**. It addresses a critical problem in agriculture: the lack of quick, accessible diagnostic tools for detecting crop diseases. Arecanut (*Areca catechu*) is a valuable cash crop in tropical regions, but it suffers from severe fungal diseases like fruit rot (Koleroga), which can cause up to 90% yield loss if not caught early.
    
    ### 🔬 Deep Learning & Transfer Learning
    The system utilizes three state-of-the-art Convolutional Neural Network (CNN) architectures pre-trained on the ImageNet database:
    1. **EfficientNet-B0**: Balances parameter size and classification accuracy using compound coefficient scaling.
    2. **MobileNetV3**: Designed for low-power edge deployment, optimized using hardware-aware search.
    3. **ResNet50**: Resolves gradient vanishing issues in deep layers using residual skip connections.
    
    By freezing the convolutional feature extraction backbones and fine-tuning dense classification layers, we achieve high accuracy with significantly smaller training sizes.
    """)

    # 14-Class Taxonomy Table
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
    
    # Future Scope Section
    st.write("---")
    st.markdown("### 🚀 Future Scope & Scalability")
    st.write("The software architecture is modularized to support future integrations:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        * **Localization**: Integrating YOLOv8/v11 for bounding-box disease localization.
        * **Severity Estimation**: Quantifying disease progression on leaf surfaces.
        * **Mobile Apps**: Compiling a Flutter/Android app connecting to this API.
        * **Multilingual Support**: Kannada language translation & voice feedback.
        """)
        
    with col2:
        st.markdown("""
        * **Drone Monitoring**: Processing multi-spectral aerial images of plantation crowns.
        * **IoT Integration**: Weather-based disease forecasting using temperature/moisture sensors.
        * **Offline Mode**: Running lightweight models offline on mobile devices.
        * **GPS Mapping**: Marking diseased trees on a map for localized spraying.
        """)

