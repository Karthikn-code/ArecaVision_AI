# System Architecture & Design Document

## 1. Executive Summary
**ArecaVision AI** is a state-of-the-art decision-support tool developed to automatically diagnose conditions and diseases in areca nut palms. The architecture uses deep Convolutional Neural Networks (CNNs) fine-tuned via Transfer Learning on a custom structured agricultural database, visualizes model focus zones using Grad-CAM, maintains transaction history in an SQLite database, and packages reports as downloadable PDF diagnostic sheets.

---

## 2. Layered Architecture Design

The system is developed using a layered engineering architecture to ensure strict separation of concerns, scalability, and code reuse:

```
┌────────────────────────────────────────────────────────┐
│                   Presentation Layer                   │
│               - Streamlit app.py (Entry)               │
│               - pages/ (Home, Detect, History, etc.)   │
└───────────────────────────┬────────────────────────────┘
                            │ (Control Routing & Input)
                            v
┌────────────────────────────────────────────────────────┐
│                  Business Logic Layer                  │
│               - data_prep.py (Split / MD5 check)       │
│               - augmentor.py (OpenCV bilateral denoise)│
│               - model_registry.py & architectures.py    │
│               - gradcam.py (GradientTape activations)  │
│               - pdf_generator.py (fpdf2 layout engine)  │
└───────────────────────────┬────────────────────────────┘
                            │ (Query execution & JSON lookup)
                            v
┌────────────────────────────────────────────────────────┐
│                   Data Access Layer                    │
│               - db_manager.py (SQLite SQLite3)         │
│               - engine.py (JSON Facts parser)          │
└───────────────────────────┬────────────────────────────┘
                            │ (Local reads & writes)
                            v
┌────────────────────────────────────────────────────────┐
│                     Storage Layer                      │
│               - results/areca_health.db (SQLite)       │
│               - recommendation/disease_database.json   │
│               - results/saved_models/*.keras           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema (SQLite)

Prediction records are logged into a local relational database `results/areca_health.db` under the table `prediction_history`:

```sql
CREATE TABLE prediction_history (
    prediction_id TEXT PRIMARY KEY,    -- Auto-generated UUID or slice
    image_path TEXT NOT NULL,          -- Path to preprocessed image on disk
    predicted_class TEXT NOT NULL,     -- Classification output category
    confidence REAL NOT NULL,          -- Float value [0.0 - 1.0]
    date TEXT NOT NULL,                -- YYYY-MM-DD
    time TEXT NOT NULL,                -- HH:MM:SS
    processing_time REAL NOT NULL,     -- Float in seconds
    model_used TEXT NOT NULL           -- Model name (e.g. EfficientNet-B0)
);
```

---

## 4. Preprocessing & Denoising Pipeline

Raw agricultural images captured in field conditions frequently contain dust, shadows, and varying exposures. The preprocessing module implements two critical steps:
1. **Denoising (Bilateral Filtering)**:
   Unlike standard Gaussian blur which washes out fine details, Bilateral Filtering uses both a spatial domain Gaussian and a radiometric range Gaussian. It averages pixels only if they are close in space and similar in intensity, successfully eliminating background noise while keeping leaf lesions and spot edges sharp.
2. **Normalization**:
   Inputs are resized to $224 \times 224 \times 3$ and rescaled to $[0, 1]$ range:
   $$X_{\text{norm}} = \frac{X}{255.0}$$

---

## 5. Explainable AI: Grad-CAM Mathematics

Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of the score for a target class flowing into the final convolutional layer of a CNN to produce a coarse localization map highlighting the regions of the image that influenced the classification decision.

### Mathematical Formulation:
1. **Score Gradient Calculation**:
   Let the class score for the target class $c$ be $y^c$ (before the softmax layer). Let $A^k$ represent the feature map activation of the $k$-th channel of the final convolutional layer. The gradient of the score $y^c$ with respect to the feature map $A^k$ is:
   $$\frac{\partial y^c}{\partial A^k}$$

2. **Global Average Pooling for Weights**:
   The importance weight $\alpha_k^c$ for the $k$-th feature map channel is calculated by average pooling the gradients over the spatial dimensions (height $u$ and width $v$):
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{u} \sum_{j=1}^{v} \frac{\partial y^c}{\partial A_{i, j}^k}$$
   where $Z = u \times v$ is the spatial area of the feature map.

3. **Weighted Linear Combination & ReLU**:
   We compute a weighted combination of forward activation maps and follow with a Rectified Linear Unit (ReLU) to isolate positive features (visual regions that increase the target score, discarding those that decrease it):
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

4. **Overlay Generation**:
   The resulting 2D heatmap $L_{\text{Grad-CAM}}^c$ is normalized, upsampled to the original image dimensions ($224 \times 224$), colored using a pseudo-color map (Jet), and superimposed on the original image with transparency factor $\alpha = 0.5$.
