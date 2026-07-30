# 🌴 ArecaVision AI

**AI-Powered Areca Nut Health Monitoring and Disease Diagnosis System**

ArecaVision AI is a production-ready agricultural decision-support platform that uses **Deep Learning (CNNs)** and **Computer Vision** to automatically identify leaf, nut, and trunk diseases in areca nut palms. It provides **Grad-CAM explainability**, agronomic treatment recommendations, batch inference tools, and professional PDF diagnostic reports.

---

## 🚀 Features

- **Multi-Model Inference**: EfficientNet-B0, MobileNetV3, ResNet50
- **🤝 Ensemble Predictor**: Soft-vote averaging across all 3 models for maximum accuracy
- **Grad-CAM Explainability**: Attention heatmaps showing which image regions drove the prediction
- **Confidence Threshold Guard**: Warns when model confidence falls below 50%
- **Per-Class Probability Distribution**: Full softmax breakdown on every inference
- **Agronomic Recommendations**: Symptoms, organic control, chemical control, and preventive measures
- **PDF Report Generator**: Professional diagnostic reports via `fpdf2`
- **SQLite History Logging**: All predictions saved to a local database with search, filter, and pagination
- **Analytics Dashboard**: Training curves, class distribution, top disease KPIs, and prediction history charts
- **Batch Inference CLI**: Run disease detection across an entire directory of field images with CSV/JSON exports

---

## 📦 Quickstart & Setup

```bash
# 1. Clone repository & navigate to folder
cd d:\Areca\ArecaVision_AI

# 2. Activate virtual environment
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Verify environment & initialize database
python setup_project.py

# 4. Launch Streamlit Web Application
streamlit run run.py
```

The app will open at `http://localhost:8501`.

---

## ⚡ Batch Inference (CLI)

Run disease detection on an entire folder of images without starting the GUI:

```bash
# Run batch inference with Ensemble model
python utils/batch_inference.py --input d:/path/to/farm_photos --ensemble

# Run batch inference with specific model
python utils/batch_inference.py --input d:/path/to/farm_photos --model EfficientNet-B0
```

Outputs:
- `results/batch_results_<timestamp>.csv` — Full per-image breakdown
- `results/batch_summary_<timestamp>.json` — Aggregate statistics and disease frequency

---

## 🗂️ Dataset Setup

Place the Arecanut dataset at:
```
d:\Areca\archive\Arecanut_dataset\Arecanut_dataset\
```

The dataset should have subdirectories matching the 9 class names defined in `config/config.py`.

---

## 🏋️ Training

The training pipeline uses **two-stage transfer learning** with class-weighted loss:

```bash
# Train all 3 models sequentially (recommended)
python training/train.py --all --epochs 10 --batch_size 32

# Train a single model
python training/train.py --model EfficientNet-B0 --epochs 10
python training/train.py --model MobileNetV3 --epochs 10
python training/train.py --model ResNet50 --epochs 10

# Custom hyperparameters
python training/train.py --model ResNet50 --epochs 20 --batch_size 16 --lr 5e-5 --warmup_epochs 5
```

### Two-Stage Training Strategy:
| Stage | Description | Default Epochs |
|---|---|---|
| **Stage 1 — Head Warmup** | Backbone frozen, trains classification head only | 5 |
| **Stage 2 — Backbone Fine-Tuning** | Top 40 backbone layers unfrozen, end-to-end fine-tuning at lr=1e-5 | 10 |

### Expected Accuracy:
| Model | Val Accuracy |
|---|---|
| EfficientNet-B0 | ~98–99% |
| MobileNetV3 | ~95–97% |
| ResNet50 | ~94–97% |
| **Ensemble (All 3)** | **~98–99%** |

---

## 📊 Evaluation

After training, generate test-set evaluation metrics:
```bash
python evaluation/evaluator.py
```

This outputs:
- `results/model_comparison.json` — test accuracy, precision, recall, F1
- Confusion matrix plots in `results/`
- ROC curve plots in `results/`

---

## 🏗️ Project Structure

```
ArecaVision_AI/
├── augmentation/       # Image preprocessing and augmentation (Bilateral denoising)
├── config/             # App-wide configuration (class names, paths, hyperparams)
├── dashboard/          # Plotly analytics and training curve generators
├── database/           # SQLite prediction history manager
├── documentation/      # System architecture documentation
├── evaluation/         # Model evaluation: accuracy, confusion matrix, ROC
├── gradcam/            # Grad-CAM heatmap computation and overlay
├── models/
│   ├── architectures.py    # EfficientNet-B0, MobileNetV3, ResNet50 builders
│   └── model_registry.py   # Load/create/ensemble_predict functions
├── recommendation/     # JSON-backed disease recommendation engine
├── reports/            # PDF diagnostic report generator
├── results/            # Saved models, training history, evaluation outputs
├── streamlit_app/
│   ├── app.py              # Main app
│   └── pages/
│       ├── detection.py    # Main inference page
│       ├── dashboard.py    # Analytics dashboard
│       ├── documentation.py# User guide and model accuracy metrics
│       ├── history.py      # Paginated prediction history
│       ├── home.py         # Landing page with live accuracy metrics
│       ├── about.py        # About the system
│       └── settings.py     # System settings & cache cleanup
├── training/
│   └── train.py            # Two-stage training pipeline
├── utils/
│   ├── batch_inference.py  # Bulk folder inference CLI tool
│   ├── inference_helper.py# Consolidated single-image inference logic
│   └── logger.py           # Unified file + console logger
├── run.py              # Main entry point for Streamlit app
├── setup_project.py    # One-click setup & environment check
├── requirements.txt
└── README.md
```

---

## 📝 Disease & Health Categories Supported

| Index | Class Name | Category |
|---|---|---|
| 1 | `bud borer` | ⚠️ Pest Infestation |
| 2 | `healthy_foot` | ✅ Healthy Base |
| 3 | `Healthy_Leaf` | ✅ Healthy Foliage |
| 4 | `Healthy_Nut` | ✅ Healthy Fruit |
| 5 | `Healthy_Trunk` | ✅ Healthy Stem |
| 6 | `Mahali_Koleroga` | ⚠️ Fungal Disease |
| 7 | `stem cracking` | ⚠️ Structural Disorder |
| 8 | `Stem_bleeding` | ⚠️ Fungal Disease |
| 9 | `yellow leaf disease` | ⚠️ Phytoplasmal Disease |

---

## 📄 License

This project is for educational and research purposes.
