import pandas as pd
import numpy as np

class OutlierDetector:
    """
    Detect outliers in keypoint trajectories using either:
     - derivative thresholds
     - rolling median thresholds
     - any other method
    """

    def __init__(self, diff_threshold=2.5, rolling_window=20, rolling_k=2.5):
        self.diff_threshold = diff_threshold
        self.rolling_window = rolling_window
        self.rolling_k = rolling_k

    def detect_outliers_diff(self, df: pd.DataFrame, joint_pairs) -> set:
        """
        Return set of row indices that are outliers based on 
        large derivative (frame-to-frame changes).
        """
        # Implementation from your code
        pass

    def detect_outliers_rolling(self, df: pd.DataFrame, joint_pairs) -> set:
        """
        Return set of row indices that are outliers based on rolling median filter.
        """
        # Implementation from your code
        pass

    def combine_outliers(self, df, joint_pairs) -> set:
        """
        Combine the outlier detection approaches (if needed).
        """
        diff_set = self.detect_outliers_diff(df, joint_pairs)
        rolling_set = self.detect_outliers_rolling(df, joint_pairs)
        return diff_set.union(rolling_set)
