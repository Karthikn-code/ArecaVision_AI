"""
setup_project.py
----------------
One-time project setup and environment verification script.
Run this before first use to ensure all required directories exist,
the SQLite database is initialized, and dependencies are available.

Usage:
    python setup_project.py
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.config import (
    BASE_DIR, RESULTS_DIR, SAVED_MODELS_DIR, SPLIT_DATASET_DIR, DB_PATH
)

REQUIRED_DIRS = [
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    SPLIT_DATASET_DIR,
    os.path.join(RESULTS_DIR, "temp"),
    os.path.join(BASE_DIR, "config"),
    os.path.join(BASE_DIR, "database"),
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "augmentation"),
    os.path.join(BASE_DIR, "preprocessing"),
    os.path.join(BASE_DIR, "gradcam"),
    os.path.join(BASE_DIR, "recommendation"),
    os.path.join(BASE_DIR, "reports"),
    os.path.join(BASE_DIR, "evaluation"),
    os.path.join(BASE_DIR, "training"),
    os.path.join(BASE_DIR, "dashboard"),
    os.path.join(BASE_DIR, "utils"),
    os.path.join(BASE_DIR, "documentation"),
    os.path.join(BASE_DIR, "streamlit_app"),
    os.path.join(BASE_DIR, "streamlit_app", "pages"),
]

def check_dependencies():
    """Verify all required Python packages are importable."""
    packages = {
        "tensorflow": "tensorflow",
        "streamlit": "streamlit",
        "cv2": "opencv-python",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "sklearn": "scikit-learn",
        "fpdf": "fpdf2",
        "PIL": "Pillow"
    }
    missing = []
    for module_name, pip_name in packages.items():
        try:
            __import__(module_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} — MISSING")
            missing.append(pip_name)
    return missing


def create_directories():
    """Ensure all required project directories exist."""
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  📁 {os.path.relpath(d, BASE_DIR)}")


def initialize_database():
    """Create or verify the SQLite database schema."""
    from database.db_manager import init_db
    init_db()
    db_size = os.path.getsize(DB_PATH) / 1024.0 if os.path.exists(DB_PATH) else 0.0
    print(f"  🗄️  Database at {DB_PATH} ({db_size:.1f} KB)")


def verify_recommendation_engine():
    """Verify the disease database JSON loads correctly."""
    from recommendation.engine import RecommendationEngine
    engine = RecommendationEngine()
    count = len(engine.database)
    print(f"  📋 Recommendation database: {count} disease entries loaded")
    return count > 0


def main():
    print("=" * 60)
    print("  ArecaVision AI — Project Setup & Environment Verification")
    print("=" * 60)

    print("\n[1/4] Checking Python packages...")
    missing = check_dependencies()

    if missing:
        print(f"\n⚠️  Missing packages detected: {', '.join(missing)}")
        print(f"     Install with: pip install {' '.join(missing)}")
    else:
        print("     All packages OK.\n")

    print("[2/4] Creating required directories...")
    create_directories()
    print("     Directories OK.\n")

    print("[3/4] Initializing SQLite database...")
    try:
        initialize_database()
        print("     Database OK.\n")
    except Exception as e:
        print(f"  ❌ Database initialization failed: {e}\n")

    print("[4/4] Verifying recommendation engine...")
    try:
        ok = verify_recommendation_engine()
        if ok:
            print("     Recommendation engine OK.\n")
        else:
            print("  ⚠️  Recommendation database appears empty.\n")
    except Exception as e:
        print(f"  ❌ Recommendation engine error: {e}\n")

    print("=" * 60)
    if not missing:
        print("\n✅  Setup complete! You can now launch the app with:")
        print("\n     streamlit run run.py\n")
        print("  Or train models with:")
        print("     python training/train.py --all\n")
    else:
        print("\n⚠️  Setup incomplete. Install missing packages first.\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
