import os

# Centralized Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAVED_MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")
SPLIT_DATASET_DIR = os.path.join(RESULTS_DIR, "split_dataset")

# Database Configuration
DB_PATH = os.path.join(RESULTS_DIR, "areca_health.db")

# Image Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Dataset Split Configuration
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# Disease Categories (14 Classes — MUST match TF dataset alphabetical directory sort order)
# tf.keras.utils.image_dataset_from_directory uses case-sensitive sort (uppercase before lowercase).
# Verified actual order by running: dataset.class_names on the split dataset.
CLASS_NAMES = [
    "Arecanut_YellowBrownSpot",   # index 0  (TF sorted: A > C > H > H > H > M > S > W > W > W > b > h > s > y)
    "CCI_Caterpillars",            # index 1
    "Healthy_Leaf",                # index 2
    "Healthy_Nut",                 # index 3
    "Healthy_Trunk",               # index 4
    "Mahali_Koleroga",             # index 5
    "Stem_bleeding",               # index 6
    "WCLWD_DryingofLeaflets",      # index 7
    "WCLWD_Flaccidity",            # index 8
    "WCLWD_Yellowing",             # index 9
    "bud borer",                   # index 10
    "healthy_foot",                # index 11
    "stem cracking",               # index 12
    "yellow leaf disease",         # index 13
]

# Display Names for UI
DISPLAY_NAMES = {
    "bud borer": "Bud Borer (Pest)",
    "healthy_foot": "Healthy Foot / Base",
    "Healthy_Leaf": "Healthy Leaf",
    "Healthy_Nut": "Healthy Nut",
    "Healthy_Trunk": "Healthy Trunk",
    "Mahali_Koleroga": "Mahali / Koleroga (Fruit Rot)",
    "stem cracking": "Stem Cracking (Disorder)",
    "Stem_bleeding": "Stem Bleeding (Disease)",
    "yellow leaf disease": "Yellow Leaf Disease",
    "Arecanut_YellowBrownSpot": "Yellow Brown Leaf Spot (Fungal)",
    "CCI_Caterpillars": "Caterpillar Foliage Infestation (Pest)",
    "WCLWD_DryingofLeaflets": "Leaf Wilt / WCLWD (Drying Stage)",
    "WCLWD_Flaccidity": "Leaf Wilt / WCLWD (Drooping Stage)",
    "WCLWD_Yellowing": "Leaf Wilt / WCLWD (Yellowing Stage)"
}

# Augmentation Parameters
AUGMENTATION_PARAMS = {
    "rotation_range": 15,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.15,
    "brightness_range": (0.85, 1.15),
    "contrast_range": (0.85, 1.15)
}

# Recommendation JSON Path
RECOMMENDATION_JSON_PATH = os.path.join(BASE_DIR, "recommendation", "disease_database.json")

# Object Detection Configuration
DETECTION_V8I_PATH = os.path.join(os.path.dirname(BASE_DIR), "arecanut leaf disease.v8i.tensorflow")
DETECTION_V2I_PATH = os.path.join(os.path.dirname(BASE_DIR), "Arecanut leaf disease detection.v2i.tensorflow")
DETECTION_CONF_THRESHOLD = 0.40
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_CLASSES = ["disease-leaf", "non-diseased-leaf", "DiseaseSpot", "HealthySpot"]

