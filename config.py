"""
Configuration module for MABe Mouse Behavior Detection
"""
from pathlib import Path
from typing import List


class Config:
    """Centralized configuration"""
    
    # Base paths
    MABE_PKG_DIR = Path("/kaggle/input/mabe-package")
    STARTER_DIR = Path("/kaggle/input/mabe-starter-train-ja")
    INPUT_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
    
    # Data paths
    TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
    TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
    TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

    # Working paths
    WORKING_DIR = Path("/kaggle/working")
    SELF_FEATURE_DIR = WORKING_DIR / "self_features"
    PAIR_FEATURE_DIR = WORKING_DIR / "pair_features"
    MODEL_DIR = WORKING_DIR / "results"

    # Index columns
    INDEX_COLS = [
        "video_id",
        "agent_mouse_id",
        "target_mouse_id",
        "video_frame",
    ]

    # Body parts
    BODY_PARTS = [
        "ear_left", "ear_right", "nose", "neck", "body_center",
        "lateral_left", "lateral_right", "hip_left", "hip_right",
        "tail_base", "tail_tip",
    ]

    # Self behaviors
    SELF_BEHAVIORS = [
        "biteobject", "climb", "dig", "exploreobject", "freeze",
        "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom",
    ]

    # Pair behaviors
    PAIR_BEHAVIORS = [
        "allogroom", "approach", "attack", "attemptmount", "avoid",
        "chase", "chaseattack", "defend", "disengage", "dominance",
        "dominancegroom", "dominancemount", "ejaculate", "escape",
        "flinch", "follow", "intromit", "mount", "reciprocalsniff",
        "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
        "submit", "tussle",
    ]

    # Temporal feature windows
    TEMPORAL_WINDOWS = [3, 5, 10, 15, 30]
    
    # Post-processing parameters
    MIN_INTERVAL_LENGTH = 3
    MERGE_GAP_THRESHOLD = 5
    SMOOTHING_WINDOW = 5
    
    # Ensemble parameters
    USE_WEIGHTED_ENSEMBLE = True
    FOLD_WEIGHT_DECAY = 0.9
    
    # Confidence threshold
    MIN_CONFIDENCE = 0.3
    
    # Behavior-specific thresholds
    BEHAVIOR_THRESHOLD_MULTIPLIER = {
        "attack": 0.9,
        "chase": 0.95,
        "sniff": 1.0,
        "approach": 1.0,
        "mount": 0.85,
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.WORKING_DIR.mkdir(parents=True, exist_ok=True)
        cls.SELF_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        cls.PAIR_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_inputs(cls):
        """Validate that required input directories exist"""
        if not cls.STARTER_DIR.exists():
            raise FileNotFoundError(
                "Dataset 'mabe-starter-train-ja' is not attached. "
                "Click 'Add input' and add it before running."
            )
        
        if not cls.MABE_PKG_DIR.exists():
            raise FileNotFoundError(
                "Dataset 'mabe-package' is not attached."
            )

