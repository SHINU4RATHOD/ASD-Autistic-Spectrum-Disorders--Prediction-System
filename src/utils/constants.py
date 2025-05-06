# src/utils/constants.py
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed")
TEST_DATA_DIR = os.path.join(DATA_ROOT, "test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

# Features
SELECTED_FEATURES = [
    "arm_flapping", "left_flap_count", "right_flap_count", "mean_flap_rate", "mean_flap_power",
    "head_bang_count", "head_bang_rate", "body_rocking", "rocking_frequency",
    "spinning", "spin_ratio", "spin_velocity", "wrist_variability_diff",
    "mean_wrist_nose_distance", "wrist_nose_variability", "elbow_angle_variability"
]

# MediaPipe to BODY_25 mapping
BODY_PARTS = {
    "Nose": 0, "MidHip": 8, "RHip": 9, "LHip": 12,
    "RShoulder": 2, "LShoulder": 5, "RElbow": 3, "LElbow": 6,
    "RWrist": 4, "LWrist": 7
}
MEDIAPIPE_TO_BODY25 = {
    0: 0,   # Nose
    12: 2,  # RShoulder
    11: 5,  # LShoulder
    14: 3,  # RElbow
    13: 6,  # LElbow
    16: 4,  # RWrist
    15: 7,  # LWrist
    24: 9,  # RHip
    23: 12, # LHip
}

# Model training
MODEL_CLASSES = {
    "LogisticRegression": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
    "RandomForestClassifier": {"n_estimators": 100, "class_weight": "balanced", "random_state": 42},
    "SVC": {"kernel": "rbf", "class_weight": "balanced", "random_state": 42},
    "KNeighborsClassifier": {"n_neighbors": 5},
    "XGBClassifier": {"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42},
    "MLPClassifier": {"hidden_layer_sizes": (100, 50), "max_iter": 1000, "random_state": 42}
}