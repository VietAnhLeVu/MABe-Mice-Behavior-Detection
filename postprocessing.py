"""
Post-processing methods for MABe Mouse Behavior Detection
"""
import numpy as np
import polars as pl
from scipy.signal import medfilt

from config import Config


class AdvancedPostProcessor:
    """Advanced post-processing to clean up predictions"""
    
    @staticmethod
    def smooth_predictions(
        predictions: np.ndarray, 
        window_size: int = None
    ) -> np.ndarray:
        """
        Apply median filter to smooth predictions
        
        Args:
            predictions: Binary predictions array
            window_size: Window size (odd number)
            
        Returns:
            Smoothed predictions
        """
        if window_size is None:
            window_size = Config.SMOOTHING_WINDOW
            
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
            
        return medfilt(predictions.astype(np.float64), kernel_size=window_size)
    
    @staticmethod
    def remove_short_intervals(
        intervals_df: pl.DataFrame,
        min_length: int = None
    ) -> pl.DataFrame:
        """
        Remove intervals that are too short (noise)
        
        Args:
            intervals_df: DataFrame with start_frame, stop_frame
            min_length: Minimum interval length
            
        Returns:
            Filtered DataFrame
        """
        if min_length is None:
            min_length = Config.MIN_INTERVAL_LENGTH
            
        return intervals_df.filter(
            (pl.col("stop_frame") - pl.col("start_frame")) >= min_length
        )
    
    @staticmethod
    def merge_nearby_intervals(
        intervals_df: pl.DataFrame,
        gap_threshold: int = None
    ) -> pl.DataFrame:
        """
        Merge nearby intervals with the same behavior
        
        Args:
            intervals_df: DataFrame with intervals
            gap_threshold: Maximum gap to merge
            
        Returns:
            Merged intervals DataFrame
        """
        if gap_threshold is None:
            gap_threshold = Config.MERGE_GAP_THRESHOLD
            
        if intervals_df.height == 0:
            return intervals_df
        
        # Group by video, agent, target, action
        groups = intervals_df.group_by(
            ["video_id", "agent_id", "target_id", "action"],
            maintain_order=True
        )
        
        merged_results = []
        
        for keys, group in groups:
            # Sort by start_frame
            sorted_group = group.sort("start_frame")
            rows = sorted_group.rows(named=True)
            
            if not rows:
                continue
                
            merged = []
            current = rows[0].copy()
            
            for row in rows[1:]:
                # Check if can merge
                gap = row["start_frame"] - current["stop_frame"]
                
                if gap <= gap_threshold:
                    # Merge: extend current interval
                    current["stop_frame"] = row["stop_frame"]
                else:
                    # Save current and start new
                    merged.append(current)
                    current = row.copy()
            
            merged.append(current)
            merged_results.extend(merged)
        
        if not merged_results:
            return intervals_df.clear()
            
        return pl.DataFrame(merged_results)
    
    @staticmethod
    def apply_behavior_specific_threshold(
        behavior: str, 
        base_threshold: float
    ) -> float:
        """
        Apply multiplier to threshold based on behavior
        
        Args:
            behavior: Behavior name
            base_threshold: Base threshold value
            
        Returns:
            Adjusted threshold
        """
        multiplier = Config.BEHAVIOR_THRESHOLD_MULTIPLIER.get(behavior, 1.0)
        return base_threshold * multiplier

