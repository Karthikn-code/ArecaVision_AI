# ArecaVision AI

**AI-Powered Areca Nut Health Monitoring and Disease Diagnosis System**

ArecaVision AI is a production-ready agricultural decision-support platform that uses **Deep Learning (CNNs)** and **Computer Vision** to automatically identify leaf, nut, and trunk diseases in areca nut palms across **14 health/disease categories**. It provides Grad-CAM explainability, agronomic treatment recommendations, batch inference tools, and professional PDF diagnostic reports.

---

## Features

- **Multi-Model Inference**: EfficientNet-B0, MobileNetV3, ResNet50
- **Ensemble Predictor**: Soft-vote averaging across all 3 models for maximum accuracy
- **Grad-CAM Explainability**: Attention heatmaps showing which image regions drove the prediction
- **14-Class Disease Taxonomy**: Covers all major arecanut pests, fungal diseases, and healthy states
- **Confidence Threshold Guard**: Warns when model confidence falls below 50%
- **Per-Class Probability Distribution**: Full softmax breakdown on every inference
- **Agronomic Recommendations**: Symptoms, organic control, chemical control, and preventive measures (English + Kannada)
- **PDF Report Generator**: Professional diagnostic reports via `fpdf2`
- **SQLite History Logging**: All predictions saved to a local database with search, filter, and pagination
- **Analytics Dashboard**: Training curves, class distribution, top disease KPIs, and prediction history charts
- **Batch Inference CLI**: Run disease detection across an entire directory of field images with CSV/JSON exports

---

## Quickstart & Setup

```bash
# 1. Clone repository & navigate to folder
git clone https://github.com/Karthikn-code/ArecaVision_AI.git
cd ArecaVision_AI

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify environment & initialize database
python setup_project.py

# 5. Launch Streamlit Web Application
streamlit run run.py
```

The app will open at `http://localhost:8501`.

---

## Training

The training pipeline uses **two-stage transfer learning** with cosine LR warmup and class-weighted loss:

```bash
# Train all 3 models sequentially (recommended, avoids OOM on CPU)
python training/train.py --all --batch_size 16 --warmup_epochs 5 --epochs 15

# Train a single model
python training/train.py --model EfficientNet-B0 --batch_size 16
python training/train.py --model MobileNetV3    --batch_size 16
python training/train.py --model ResNet50        --batch_size 16
```

### Two-Stage Training Strategy

| Stage | Description | Default |
|---|---|---|
| **Stage 1 — Head Warmup** | Backbone frozen; trains classification head with Adam + WarmupCosineDecay | 5 epochs |
| **Stage 2 — Fine-Tuning** | Top 60 backbone layers unfrozen; AdamW + EarlyStopping + ReduceLROnPlateau | 15 epochs |

> **Note**: TensorFlow >= 2.11 does not support GPU on native Windows. Use WSL2 or DirectML plugin for GPU acceleration.

---

## Evaluation

After training, generate test-set evaluation metrics:

```bash
python evaluation/evaluator.py
```

Outputs saved to `results/`:
- `model_comparison.json` — per-model accuracy, precision, recall, F1
- `*_confusion_matrix.png` — 14×14 confusion matrix plots
- `*_roc_curve.png` — per-class ROC curves

---

## Batch Inference (CLI)

```bash
# Run batch inference with Ensemble model
python utils/batch_inference.py --input /path/to/farm_photos --ensemble

# Run batch inference with specific model
python utils/batch_inference.py --input /path/to/farm_photos --model EfficientNet-B0
```

---

## Project Structure

```
ArecaVision_AI/
├── augmentation/           # Image augmentation pipeline (flip, rotation, zoom, brightness)
├── config/
│   └── config.py           # CLASS_NAMES (14, alphabetical TF sort), paths, hyperparams
├── dashboard/              # Plotly analytics and training curve generators
├── database/               # SQLite prediction history manager
├── evaluation/
│   └── evaluator.py        # Model evaluation: accuracy, confusion matrix, ROC curves
├── gradcam/
│   └── gradcam.py          # Grad-CAM heatmap computation and overlay
├── models/
│   ├── architectures.py    # EfficientNet-B0, MobileNetV3, ResNet50 builders (14-class head)
│   └── model_registry.py   # Load/create/ensemble_predict helpers
├── object_detection/       # Leaf spot detector (TF Object Detection API)
├── preprocessing/
│   └── data_prep.py        # Dataset split (70/15/15), deduplication, corruption check
├── recommendation/
│   ├── disease_database.json   # English agronomic recommendations for all 14 classes
│   ├── engine.py               # Recommendation lookup engine
│   └── translations.py         # Kannada (ಕನ್ನಡ) translations for all 14 classes
├── reports/                # PDF diagnostic report generator
├── results/
│   ├── saved_models/       # Trained .keras model files (gitignored)
│   └── split_dataset/      # Train/Val/Test splits (gitignored)
├── streamlit_app/
│   └── pages/
│       ├── detection.py    # Main inference + Grad-CAM page
│       ├── dashboard.py    # Analytics dashboard
│       ├── history.py      # Paginated prediction history
│       └── home.py         # Landing page
├── tests/                  # Unit tests (unittest)
├── training/
│   └── train.py            # Two-stage training pipeline with WarmupCosineDecay
├── utils/
│   ├── batch_inference.py  # Bulk folder inference CLI
│   ├── inference_helper.py # Single-image inference helper
│   ├── severity_estimator.py # HSV-based disease severity scorer
│   └── logger.py           # Unified file + console logger
├── run.py                  # Streamlit app entry point
├── setup_project.py        # One-click setup & environment check
├── requirements.txt
└── Dockerfile
```

---

## Disease & Health Categories (14 Classes)

> **Important**: Class indices follow TensorFlow's case-sensitive alphabetical sort of folder names.

| Index | Class Name | Display Name | Category |
|---|---|---|---|
| 0 | `Arecanut_YellowBrownSpot` | Yellow Brown Leaf Spot | ⚠️ Fungal Disease |
| 1 | `CCI_Caterpillars` | Caterpillar Foliage Infestation | ⚠️ Pest Infestation |
| 2 | `Healthy_Leaf` | Healthy Leaf | ✅ Healthy |
| 3 | `Healthy_Nut` | Healthy Nut | ✅ Healthy |
| 4 | `Healthy_Trunk` | Healthy Trunk | ✅ Healthy |
| 5 | `Mahali_Koleroga` | Mahali / Koleroga (Fruit Rot) | ⚠️ Fungal Disease |
| 6 | `Stem_bleeding` | Stem Bleeding | ⚠️ Fungal Disease |
| 7 | `WCLWD_DryingofLeaflets` | Leaf Wilt / WCLWD (Drying) | ⚠️ Phytoplasmal |
| 8 | `WCLWD_Flaccidity` | Leaf Wilt / WCLWD (Drooping) | ⚠️ Phytoplasmal |
| 9 | `WCLWD_Yellowing` | Leaf Wilt / WCLWD (Yellowing) | ⚠️ Phytoplasmal |
| 10 | `bud borer` | Bud Borer (Pest) | ⚠️ Pest Infestation |
| 11 | `healthy_foot` | Healthy Foot / Base | ✅ Healthy |
| 12 | `stem cracking` | Stem Cracking | ⚠️ Structural Disorder |
| 13 | `yellow leaf disease` | Yellow Leaf Disease | ⚠️ Phytoplasmal |

---

## Key Technical Notes

- **CLASS_NAMES order**: Must match `tf.keras.utils.image_dataset_from_directory` output which sorts folder names in **case-sensitive alphabetical order** (uppercase before lowercase). Verified with `dataset.class_names`.
- **Label smoothing**: 0.1 applied during warmup stage to prevent overconfidence.
- **Batch size**: Use `--batch_size 16` on CPU to avoid OOM. GPU users can use 32.

---

## License

This project is for educational and research purposes — Final Year AI & Data Science Project.
