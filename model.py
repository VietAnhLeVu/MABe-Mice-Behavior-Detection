"""
Model loading and prediction for MABe Mouse Behavior Detection
"""
from typing import List, Tuple, Optional
import numpy as np
import polars as pl
import xgboost as xgb

from config import Config
from utils import extract_mouse_id, mouse_id_to_string
from ensemble import ImprovedEnsemble
from postprocessing import AdvancedPostProcessor


class ModelLoader:
    """Load trained XGBoost models"""

    @staticmethod
    def load_fold_models(lab_id: str, behavior: str) -> List[Tuple[xgb.Booster, float]]:
        """
        Load all fold models for a behavior
        
        Args:
            lab_id: Lab ID
            behavior: Behavior name
            
        Returns:
            List of (model, threshold) tuples
        """
        behavior_dir = Config.MODEL_DIR / lab_id / behavior

        if not behavior_dir.exists():
            return []

        fold_dirs = sorted(behavior_dir.glob("fold_*"))
        models = []

        for fold_dir in fold_dirs:
            model_file = fold_dir / "model.json"
            threshold_file = fold_dir / "threshold.txt"

            if not model_file.exists() or not threshold_file.exists():
                continue

            with open(threshold_file, "r") as f:
                threshold = float(f.read().strip())

            model = xgb.Booster(model_file=str(model_file))
            models.append((model, threshold))

        return models


class ImprovedPredictionEngine:
    """Prediction engine with weighted ensemble and confidence voting"""

    @staticmethod
    def predict_behavior(
        feature_df: pl.DataFrame,
        models: List[Tuple[xgb.Booster, float]],
        behavior: str = None,
        use_voting: bool = True,
    ) -> np.ndarray:
        """
        Predict with improved ensemble
        
        Args:
            feature_df: Feature DataFrame
            models: List of (model, threshold) tuples
            behavior: Behavior name for threshold adjustment
            use_voting: Whether to use voting ensemble
            
        Returns:
            Prediction array
        """
        if not models:
            return np.zeros(feature_df.height, dtype=np.float32)

        # Get feature names from model to ensure match
        model_feature_names = models[0][0].feature_names
        
        # Only keep features that model was trained on
        available_features = [f for f in model_feature_names if f in feature_df.columns]
        
        if len(available_features) != len(model_feature_names):
            # If missing features, add columns with 0 values
            missing_features = [f for f in model_feature_names if f not in feature_df.columns]
            for mf in missing_features:
                feature_df = feature_df.with_columns(pl.lit(0.0).alias(mf))
        
        # Order features according to model
        feature_df_filtered = feature_df.select(model_feature_names)
        
        dtest = xgb.DMatrix(
            feature_df_filtered.to_pandas(), 
            feature_names=model_feature_names
        )

        predictions = []
        thresholds = []

        for model, threshold in models:
            probs = model.predict(dtest)
            
            # Apply behavior-specific threshold
            if behavior:
                threshold = AdvancedPostProcessor.apply_behavior_specific_threshold(
                    behavior, threshold
                )
            
            predictions.append(probs)
            thresholds.append(threshold)

        if use_voting:
            # Use confidence voting
            final_pred, confidence = ImprovedEnsemble.confidence_voting(
                predictions, thresholds, min_agreement=0.5
            )
            return final_pred.astype(np.float32)
        else:
            # Use weighted average
            return ImprovedEnsemble.weighted_average(
                predictions, thresholds
            ).astype(np.float32)

    @staticmethod
    def select_best_behavior(prediction_df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Select behavior with highest probability for each frame
        
        Args:
            prediction_df: DataFrame with predictions
            
        Returns:
            DataFrame with selected behaviors
        """
        pred_cols = prediction_df.select(pl.exclude(Config.INDEX_COLS)).columns

        if not pred_cols:
            return None

        result = prediction_df.with_columns(
            pl.struct(pl.col(pred_cols))
            .map_elements(
                lambda row: (
                    "none"
                    if sum(row.values()) == 0 or max(row.values()) < Config.MIN_CONFIDENCE
                    else pred_cols[int(np.argmax(list(row.values())))].split("_")[0]
                ),
                return_dtype=pl.String,
            )
            .alias("prediction")
        ).select(Config.INDEX_COLS + ["prediction"])

        return result

    @staticmethod
    def convert_to_intervals(
        frame_predictions: pl.DataFrame,
        agent_mouse_id: int,
        target_mouse_id: int,
        apply_postprocessing: bool = True,
    ) -> pl.DataFrame:
        """
        Convert frame-by-frame predictions to intervals with post-processing
        
        Args:
            frame_predictions: Frame-level predictions
            agent_mouse_id: Agent mouse ID
            target_mouse_id: Target mouse ID
            apply_postprocessing: Whether to apply post-processing
            
        Returns:
            Intervals DataFrame
        """
        intervals = (
            frame_predictions
            .filter(pl.col("prediction") != pl.col("prediction").shift(1))
            .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))
            .filter(pl.col("prediction") != "none")
            .select(
                [
                    pl.col("video_id"),
                    pl.lit(mouse_id_to_string(agent_mouse_id)).alias("agent_id"),
                    pl.lit(mouse_id_to_string(target_mouse_id)).alias("target_id"),
                    pl.col("prediction").alias("action"),
                    pl.col("video_frame").alias("start_frame"),
                    pl.col("stop_frame"),
                ]
            )
        )

        # Apply post-processing
        if apply_postprocessing and intervals.height > 0:
            # Remove short intervals
            intervals = AdvancedPostProcessor.remove_short_intervals(intervals)
            
            # Merge nearby intervals
            intervals = AdvancedPostProcessor.merge_nearby_intervals(intervals)

        return intervals

