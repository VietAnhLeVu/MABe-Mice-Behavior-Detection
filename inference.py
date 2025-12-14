"""
Inference pipeline for MABe Mouse Behavior Detection
"""
from typing import Optional
import polars as pl
from tqdm.auto import tqdm

from config import Config
from utils import extract_mouse_id
from features import FeatureLoader
from model import ModelLoader, ImprovedPredictionEngine


class ImprovedInferencePipeline:
    """Main inference orchestrator with improvements"""

    @staticmethod
    def predict_for_group(
        lab_id: str,
        video_id: int,
        agent: str,
        target: str,
        group_behaviors: pl.DataFrame,
        use_voting: bool = True,
    ) -> Optional[pl.DataFrame]:
        """
        Run inference for a single group
        
        Args:
            lab_id: Lab ID
            video_id: Video ID
            agent: Agent mouse
            target: Target mouse
            group_behaviors: Behaviors to predict for this group
            use_voting: Whether to use voting ensemble
            
        Returns:
            Submission DataFrame for this group
        """
        index_df, feature_df = FeatureLoader.load_features_for_group(
            lab_id, video_id, agent, target
        )

        if feature_df is None or feature_df.height == 0:
            return None

        prediction_df = index_df.clone()
        unique_behaviors = group_behaviors.select("behavior").unique()["behavior"].to_list()

        for behavior in unique_behaviors:
            models = ModelLoader.load_fold_models(lab_id, behavior)

            if not models:
                continue

            # Use improved prediction
            predictions = ImprovedPredictionEngine.predict_behavior(
                feature_df, models, behavior=behavior, use_voting=use_voting
            )
            
            col_name = f"{behavior}_pred"
            prediction_df = prediction_df.with_columns(
                pl.Series(name=col_name, values=predictions, dtype=pl.Float32)
            )

        frame_predictions = ImprovedPredictionEngine.select_best_behavior(prediction_df)

        if frame_predictions is None:
            return None

        agent_mouse_id = extract_mouse_id(agent)
        target_mouse_id = extract_mouse_id(target)

        intervals = ImprovedPredictionEngine.convert_to_intervals(
            frame_predictions,
            agent_mouse_id,
            target_mouse_id,
            apply_postprocessing=True,
        )

        return intervals

    @staticmethod
    def run_full_inference(
        test_df: pl.DataFrame,
        behavior_df: pl.DataFrame,
        robustify_func,
        use_voting: bool = True,
    ) -> pl.DataFrame:
        """
        Run inference on all test data
        
        Args:
            test_df: Test metadata
            behavior_df: Behavior table
            robustify_func: Post-processing function
            use_voting: Whether to use voting ensemble
            
        Returns:
            Final submission DataFrame
        """
        print("Running IMPROVED inference on all groups...")
        print(f"  Using voting: {use_voting}")
        print(f"  Min interval length: {Config.MIN_INTERVAL_LENGTH}")
        print(f"  Merge gap threshold: {Config.MERGE_GAP_THRESHOLD}")

        groups = list(
            behavior_df.group_by(
                "lab_id", "video_id", "agent", "target", maintain_order=True
            )
        )

        print(f"  ✓ Found {len(groups)} groups to process")

        group_submissions = []

        for (lab_id, video_id, agent, target), group in tqdm(
            groups, desc="Predicting groups", total=len(groups)
        ):
            group_submission = ImprovedInferencePipeline.predict_for_group(
                lab_id=lab_id,
                video_id=video_id,
                agent=agent,
                target=target,
                group_behaviors=group,
                use_voting=use_voting,
            )

            if group_submission is not None and group_submission.height > 0:
                group_submissions.append(group_submission)

        if not group_submissions:
            raise RuntimeError("No predictions generated!")

        submission = pl.concat(group_submissions, how="vertical").sort(
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "start_frame",
            "stop_frame",
        )

        print(f"  ✓ Initial submission: {submission.height} rows")

        # Apply robustify
        print("Applying robustify post-processing...")
        submission = robustify_func(submission, test_df, train_test="test")

        # Final filter
        submission = submission.filter(pl.col("start_frame") < pl.col("stop_frame"))

        print(f"  ✓ Final submission: {submission.height} rows")

        return submission

