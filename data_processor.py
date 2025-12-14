"""
Data processing for MABe Mouse Behavior Detection
"""
import polars as pl

from utils import parse_behaviors_safely


class DataProcessor:
    """Process and prepare data"""

    @staticmethod
    def build_behavior_table(test_df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert behaviors_labeled column to normalized table
        
        Args:
            test_df: Test metadata DataFrame
            
        Returns:
            Normalized behavior DataFrame
        """
        print("Building behavior table from behaviors_labeled...")

        behavior_df = (
            test_df
            .filter(pl.col("behaviors_labeled").is_not_null())
            .select(["lab_id", "video_id", "behaviors_labeled"])
            .with_columns(
                pl.col("behaviors_labeled")
                .map_elements(
                    parse_behaviors_safely,
                    return_dtype=pl.List(pl.Utf8),
                )
                .alias("behaviors_list")
            )
            .explode("behaviors_list")
            .rename({"behaviors_list": "behavior_element"})
            .with_columns(
                [
                    pl.col("behavior_element")
                    .str.split(",")
                    .list.get(0)
                    .str.replace_all("[()' ]", "")
                    .alias("agent"),
                    pl.col("behavior_element")
                    .str.split(",")
                    .list.get(1)
                    .str.replace_all("[()' ]", "")
                    .alias("target"),
                    pl.col("behavior_element")
                    .str.split(",")
                    .list.get(2)
                    .str.replace_all("[()' ]", "")
                    .alias("behavior"),
                ]
            )
            .select(["lab_id", "video_id", "agent", "target", "behavior"])
        )

        print(f"  ✓ Found {behavior_df.height} behavior instances")
        return behavior_df

