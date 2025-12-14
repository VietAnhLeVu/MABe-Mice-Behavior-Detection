"""
Main execution script for MABe Mouse Behavior Detection
Run this file on Kaggle to generate submission.csv
"""
import sys
import shutil
import warnings
import polars as pl

warnings.filterwarnings("ignore")

# ============ SETUP ============
print("=" * 60)
print("MABe Social Behavior Detection - IMPROVED Pipeline")
print("=" * 60)

print("\n[0/5] Setting up environment...")

# Import config and validate
from config import Config

# Validate inputs exist
Config.validate_inputs()
Config.setup_directories()

# Copy required files from starter dataset
print("  Copying helper files...")
shutil.copy(Config.STARTER_DIR / "self_features.py", Config.WORKING_DIR)
shutil.copy(Config.STARTER_DIR / "pair_features.py", Config.WORKING_DIR)
shutil.copy(Config.STARTER_DIR / "robustify.py", Config.WORKING_DIR)
shutil.copytree(Config.STARTER_DIR / "results", Config.MODEL_DIR, dirs_exist_ok=True)
print("  ✓ Helper files copied")

# Add working dir to path
sys.path.insert(0, str(Config.WORKING_DIR))

# ============ LOAD HELPER MODULES ============
print("\n[1/5] Loading helper modules...")

# Load external functions
exec(open(Config.WORKING_DIR / "self_features.py").read(), globals())
exec(open(Config.WORKING_DIR / "pair_features.py").read(), globals())
exec(open(Config.WORKING_DIR / "robustify.py").read(), globals())

print("  ✓ Loaded: make_self_features, make_pair_features, robustify")

# ============ LOAD DATA ============
print("\n[2/5] Loading test metadata...")
test_df = pl.read_csv(Config.INPUT_DIR / "test.csv")
print(f"  ✓ Loaded {test_df.height} test videos")

# ============ PARSE BEHAVIORS ============
print("\n[3/5] Processing behavior annotations...")
from data_processor import DataProcessor
behavior_df = DataProcessor.build_behavior_table(test_df)

# ============ GENERATE FEATURES ============
print("\n[4/5] Generating features...")
from features import FeatureGenerator

FeatureGenerator.generate_all_features(
    test_df=test_df,
    make_self_features_func=make_self_features,
    make_pair_features_func=make_pair_features,
    enhance_temporal=False,  # OFF because model not trained with temporal features
)

# ============ RUN INFERENCE ============
print("\n[5/5] Running IMPROVED inference...")
from inference import ImprovedInferencePipeline

submission = ImprovedInferencePipeline.run_full_inference(
    test_df=test_df,
    behavior_df=behavior_df,
    robustify_func=robustify,
    use_voting=True,
)

# ============ SAVE SUBMISSION ============
output_path = Config.WORKING_DIR / "submission.csv"
submission.with_row_index("row_id").write_csv(output_path)

print("\n" + "=" * 60)
print(f"✓ Submission saved to: {output_path}")
print(f"✓ Total predictions: {submission.height}")
print("=" * 60)

# Print improvement summary
print("\n📊 IMPROVEMENT SUMMARY:")
print(f"  • Temporal features: OFF (model not trained with temporal)")
print(f"  • Min interval length: {Config.MIN_INTERVAL_LENGTH} frames")
print(f"  • Merge gap threshold: {Config.MERGE_GAP_THRESHOLD} frames")
print(f"  • Weighted ensemble: {Config.USE_WEIGHTED_ENSEMBLE}")
print(f"  • Confidence voting: ON")
print(f"  • Min confidence: {Config.MIN_CONFIDENCE}")
print(f"  • Post-processing: Remove short intervals + Merge nearby")

# Preview submission
print("\n📄 Submission Preview:")
print(submission.head(10))

