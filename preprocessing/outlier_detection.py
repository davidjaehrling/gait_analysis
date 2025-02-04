import pandas as pd
import numpy as np
import cv2
from scipy.signal import butter, filtfilt

class OutlierDetector:
    """
    A simple outlier detector that flags outlier keypoint coordinates and cleans them.
    
    The detector uses two strategies to identify outliers in joint coordinate pairs:
      1. A derivative threshold: if the Euclidean difference (from one frame to the next)
         exceeds a given z-score threshold, the frame is flagged.
      2. A rolling-window approach: points deviating more than 'k' times the rolling
         standard deviation from the rolling median are flagged.
    
    Outlier rows are then set to NaN (for both x and y of each joint pair) and later
    interpolated using interpolation.
    
    Parameters (set at initialization):
      - derivative_threshold (float): z-score threshold for the derivative approach.
      - rolling_window_size (int): Window size for the rolling median/std.
      - rolling_k (float): Multiplier for the rolling standard deviation threshold.
    """
    
    def __init__(self, derivative_threshold=2.7, rolling_window_size=20, rolling_k=2.5):
        self.derivative_threshold = derivative_threshold
        self.rolling_window_size = rolling_window_size
        self.rolling_k = rolling_k

    def detect_outliers_diff(self, data, joint_pairs, threshold):
        """
        Detect outliers using the derivative approach.
        
        Parameters:
          - data: DataFrame containing the keypoint data.
          - joint_pairs: list of tuples, e.g. [('right_ankle_x', 'right_ankle_y'), ...].
          - threshold: z-score threshold.
          
        Returns:
          A set of row indices flagged as outliers.
        """
        outlier_indices = set()
        for (xcol, ycol) in joint_pairs:
            if xcol not in data.columns or ycol not in data.columns:
                continue

            # Compute derivative for each column
            dx = data[xcol].diff()
            dy = data[ycol].diff()

            # Euclidean norm of the derivatives
            dd = np.sqrt(dx**2 + dy**2)

            # Compute z-score for the derivative distances
            dd_std = dd.std()
            if dd_std != 0:
                dd_z = (dd - dd.mean()) / dd_std
            else:
                dd_z = dd * 0

            # Flag indices where the absolute z-score exceeds the threshold
            outliers = dd_z.index[dd_z.abs() > threshold].tolist()
            outlier_indices.update(outliers)
        return outlier_indices

    def detect_outliers_rolling(self, data, joint_pairs, window_size, k):
        """
        Detect outliers based on a rolling median and standard deviation.
        
        Parameters:
          - data: DataFrame containing the keypoint data.
          - joint_pairs: list of tuples, e.g. [('right_ankle_x', 'right_ankle_y'), ...].
          - window_size: Size of the rolling window.
          - k: Multiplier for the rolling standard deviation.
          
        Returns:
          A set of row indices flagged as outliers.
        """
        outlier_indices = set()
        for (xcol, ycol) in joint_pairs:
            if xcol not in data.columns or ycol not in data.columns:
                continue

            # For x coordinate:
            x_median = data[xcol].rolling(window=window_size, center=True, min_periods=1).median()
            x_std = data[xcol].rolling(window=window_size, center=True, min_periods=1).std()
            x_std_filled = x_std.bfill().ffill().fillna(0)
            x_std_filled[x_std_filled == 0] = 1e-6  # Prevent division by zero

            x_diff = (data[xcol] - x_median).abs()
            x_outliers = x_diff > (k * x_std_filled)
            outlier_indices.update(data.index[x_outliers].tolist())

            # For y coordinate:
            y_median = data[ycol].rolling(window=window_size, center=True, min_periods=1).median()
            y_std = data[ycol].rolling(window=window_size, center=True, min_periods=1).std()
            y_std_filled = y_std.bfill().ffill().fillna(0)
            y_std_filled[y_std_filled == 0] = 1e-6

            y_diff = (data[ycol] - y_median).abs()
            y_outliers = y_diff > (k * y_std_filled)
            outlier_indices.update(data.index[y_outliers].tolist())

        return outlier_indices

    def interpolate_data(self, data, columns):
        """
        Interpolate missing values in the specified columns using linear interpolation.
        
        Parameters:
          - data: DataFrame containing keypoint data.
          - columns: List of column names to interpolate.
          
        Returns:
          The DataFrame with interpolated values.
        """
        for col in columns:
            if col not in data.columns:
                continue
            data[col] = data[col].interpolate(method='linear', limit_direction='both')
        return data

    def clean(self, df):
        """
        Clean the input DataFrame by detecting and replacing outliers.
        
        Parameters:
          - df: Input DataFrame containing keypoint data.
          - startframe (int, optional): If provided, only process frames >= startframe.
          - person_idx (int): Only process data for the given person index (e.g. main subject).
          
        Returns:
          A cleaned DataFrame.
        """
        # Work on a copy of the data.
        data = df.copy()
            
        data.sort_values('frame', inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Identify joint pairs by looking for every column ending with '_x'
        # and its corresponding '_y' column.
        joint_pairs = []
        for xcol in data.columns:
            if xcol.endswith('_x'):
                base = xcol[:-2]  # remove '_x'
                ycol = base + '_y'
                if ycol in data.columns:
                    joint_pairs.append((xcol, ycol))

        # Detect outliers using the derivative method.
        diff_outliers = self.detect_outliers_diff(data, joint_pairs, self.derivative_threshold)
        # Detect outliers using the rolling-window method.
        rolling_outliers = self.detect_outliers_rolling(data, joint_pairs,
                                                        self.rolling_window_size,
                                                        self.rolling_k)
        # Combine outlier indices from both methods.
        outlier_indices = diff_outliers.union(rolling_outliers)

        print(outlier_indices)

        # For every joint pair, set the values in the outlier rows to NaN.
        for (xcol, ycol) in joint_pairs:
            data.loc[list(outlier_indices), xcol] = np.nan
            data.loc[list(outlier_indices), ycol] = np.nan

        # Also, set any values equal to zero to NaN (if zeros indicate detection failure)
        for (xcol, ycol) in joint_pairs:
            data.loc[data[xcol] == 0, xcol] = np.nan
            data.loc[data[ycol] == 0, ycol] = np.nan

        # Interpolate missing/outlier values for all joint coordinate columns.
        all_xy_cols = []
        for pair in joint_pairs:
            all_xy_cols.extend(list(pair))
        data = self.interpolate_data(data, all_xy_cols)

        # Finally, fill any remaining missing values by backward then forward filling,
        # and fill any residual NaN with zeros.
        data = data.bfill().ffill().fillna(0)

        return data

    def smooth_data(self, data, video_path, cutoff=3.0, order=2, columns=None):
        """
        Smooth the input DataFrame using a Butterworth low-pass filter.
        
        The method extracts the video's FPS from the provided video_path to compute the
        normalized cutoff frequency. It then applies a zero-phase Butterworth filter
        (via filtfilt) to each specified column.
        
        Parameters:
          - data: DataFrame containing the keypoint data (assumed to be sorted by frame).
          - video_path: Path to the video file. The FPS is extracted from this video.
          - cutoff (float): The cutoff frequency in Hz. Default is 3.0 Hz.
          - order (int): The order of the Butterworth filter. Default is 2.
          - columns (list of str, optional): List of columns to filter. If None, then all columns
            ending in '_x' or '_y' are filtered.
            
        Returns:
          A new DataFrame with the filtered (smoothed) data.
        """
        # Get the video's FPS using OpenCV.
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Compute the normalized cutoff frequency (Nyquist frequency is half the FPS).
        nyq = 0.5 * fps
        normal_cutoff = cutoff / nyq

        # Design the Butterworth filter.
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # If no specific columns are provided, select columns ending with '_x' or '_y'
        if columns is None:
            columns = [col for col in data.columns if col.endswith('_x') or col.endswith('_y')]

        # Create a copy of the data to avoid modifying the original.
        data_filtered = data.copy()

        # Apply the zero-phase Butterworth filter to each selected column.
        for col in columns:
            if not pd.api.types.is_numeric_dtype(data_filtered[col]):
                continue
            data_filtered[col] = filtfilt(b, a, data_filtered[col].values)

        return data_filtered