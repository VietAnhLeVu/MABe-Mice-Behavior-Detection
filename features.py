"""
Feature generation and enhancement for MABe Mouse Behavior Detection
"""
import gc
from typing import List
import polars as pl
from tqdm.auto import tqdm

from config import Config


class TemporalFeatureEnhancer:
    """Add temporal features to capture time information"""
    
    @staticmethod
    def add_rolling_features(df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """
        Add rolling statistics for features
        
        Args:
            df: DataFrame with original features
            feature_cols: List of feature columns
            
        Returns:
            DataFrame with added rolling features
        """
        result = df.clone()
        
        for window in Config.TEMPORAL_WINDOWS:
            for col in feature_cols[:10]:  # Top 10 features only
                # Rolling mean
                result = result.with_columns(
                    pl.col(col)
                    .rolling_mean(window_size=window, min_periods=1)
                    .alias(f"{col}_roll_mean_{window}")
                )
                
                # Rolling std
                result = result.with_columns(
                    pl.col(col)
                    .rolling_std(window_size=window, min_periods=1)
                    .fill_null(0)
                    .alias(f"{col}_roll_std_{window}")
                )
        
        return result
    
    @staticmethod
    def add_lag_features(
        df: pl.DataFrame, 
        feature_cols: List[str], 
        lags: List[int] = [1, 2, 5]
    ) -> pl.DataFrame:
        """
        Add lag features (values from previous frames)
        
        Args:
            df: Original DataFrame
            feature_cols: Columns to create lags for
            lags: Number of frames to lag
            
        Returns:
            DataFrame with lag features
        """
        result = df.clone()
        
        for lag in lags:
            for col in feature_cols[:5]:  # Top 5 features
                result = result.with_columns(
                    pl.col(col).shift(lag).fill_null(0).alias(f"{col}_lag_{lag}")
                )
                
                # Difference with lag
                result = result.with_columns(
                    (pl.col(col) - pl.col(f"{col}_lag_{lag}")).alias(f"{col}_diff_{lag}")
                )
        
        return result
    
    @staticmethod
    def add_velocity_acceleration(
        df: pl.DataFrame, 
        position_cols: List[str]
    ) -> pl.DataFrame:
        """
        Calculate velocity and acceleration from position
        """
        result = df.clone()
        
        for col in position_cols[:5]:
            # Velocity (first derivative)
            result = result.with_columns(
                (pl.col(col) - pl.col(col).shift(1))
                .fill_null(0)
                .alias(f"{col}_velocity")
            )
            
            # Acceleration (second derivative)
            result = result.with_columns(
                (pl.col(f"{col}_velocity") - pl.col(f"{col}_velocity").shift(1))
                .fill_null(0)
                .alias(f"{col}_accel")
            )
        
        return result


class FeatureGenerator:
    """Generate features from tracking data"""

    @staticmethod
    def generate_all_features(
        test_df: pl.DataFrame,
        make_self_features_func,
        make_pair_features_func,
        enhance_temporal: bool = False,
    ):
        """
        Pre-compute all features for test videos
        
        Args:
            test_df: Test metadata
            make_self_features_func: Function to create self features
            make_pair_features_func: Function to create pair features
            enhance_temporal: Whether to add temporal features
        """
        print("Generating features for all test videos...")
        print(f"  Temporal enhancement: {'ON' if enhance_temporal else 'OFF'}")

        Config.setup_directories()

        rows = test_df.rows(named=True)

        for row in tqdm(rows, desc="Computing features", total=len(rows)):
            lab_id = row["lab_id"]
            video_id = row["video_id"]

            tracking_path = Config.TEST_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
            tracking = pl.read_parquet(tracking_path)

            # Generate self features
            self_feat = make_self_features_func(metadata=row, tracking=tracking)
            
            if enhance_temporal:
                feature_cols = [c for c in self_feat.columns if c not in Config.INDEX_COLS]
                self_feat = TemporalFeatureEnhancer.add_rolling_features(self_feat, feature_cols)
                self_feat = TemporalFeatureEnhancer.add_lag_features(self_feat, feature_cols)
            
            self_feat.write_parquet(Config.SELF_FEATURE_DIR / f"{video_id}.parquet")

            # Generate pair features
            pair_feat = make_pair_features_func(metadata=row, tracking=tracking)
            
            if enhance_temporal:
                feature_cols = [c for c in pair_feat.columns if c not in Config.INDEX_COLS]
                pair_feat = TemporalFeatureEnhancer.add_rolling_features(pair_feat, feature_cols)
                pair_feat = TemporalFeatureEnhancer.add_lag_features(pair_feat, feature_cols)
            
            pair_feat.write_parquet(Config.PAIR_FEATURE_DIR / f"{video_id}.parquet")

            del self_feat, pair_feat, tracking
            gc.collect()

        print("  ✓ Features saved successfully")


class FeatureLoader:
    """Load pre-computed features"""

    @staticmethod
    def load_features_for_group(
        lab_id: str,
        video_id: int,
        agent: str,
        target: str,
    ):
        """
        Load features for a specific group
        
        Args:
            lab_id: Lab ID
            video_id: Video ID
            agent: Agent mouse
            target: Target mouse
            
        Returns:
            Tuple of (index_df, feature_df) or (None, None)
        """
        from utils import extract_mouse_id
        
        agent_mouse_id = extract_mouse_id(agent)
        target_mouse_id = extract_mouse_id(target)

        if target == "self":
            feature_path = Config.SELF_FEATURE_DIR / f"{video_id}.parquet"
            scan = pl.scan_parquet(feature_path).filter(
                pl.col("agent_mouse_id") == agent_mouse_id
            )
        else:
            feature_path = Config.PAIR_FEATURE_DIR / f"{video_id}.parquet"
            scan = pl.scan_parquet(feature_path).filter(
                (pl.col("agent_mouse_id") == agent_mouse_id)
                & (pl.col("target_mouse_id") == target_mouse_id)
            )

        full_df = scan.collect()

        if full_df.height == 0:
            return None, None

        index_df = full_df.select(Config.INDEX_COLS)
        feature_df = full_df.select(pl.exclude(Config.INDEX_COLS))

        return index_df, feature_df

