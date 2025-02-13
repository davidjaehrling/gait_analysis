import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from manual_correction.base_interactive_tool import BaseInteractiveTool

# Assuming BaseInteractiveTool is already defined as in the previous answer.
# For example:
# class BaseInteractiveTool:
#     ... (code omitted for brevity) ...

class OutlierCleaningTool(BaseInteractiveTool):
    """
    A child class of BaseInteractiveTool for cleaning outliers.
    
    Provides the following functionalities:
      - Swap left/right markers for the currently selected frame.
        The user selects one or more keypoints (e.g. ankle, knee, hip, shoulder,
        heel, big_toe) via check buttons (displayed on the right side of the figure)
        and clicks the "Swap Markers" button.
      - Delete and interpolate the coordinates for one side (right or left) for the
        selected keypoints in the current frame. Two separate buttons are provided:
        "DEL&INT Right" and "DEL&INT Left".
      - An Exit button to end the interactive session.
    
    Each operation records an undo action so that the Undo button (from the parent)
    can revert the last change.
    """
    
    def __init__(self, df: pd.DataFrame, video_path: str, skeleton_drawer,
                 y_keys=None, default_y_key=None):
        """
        Parameters are the same as in BaseInteractiveTool.
        """
        super().__init__(df, video_path, skeleton_drawer, y_keys, default_y_key)
        self._init_extra_widgets()
    
    def _init_extra_widgets(self):
        """
        Add extra widgets for outlier cleaning.  
        In this example, we reserve a portion of the figure on the right for a set of
        check buttons (to select keypoints) and add buttons for the following:
          - Swap Markers
          - DEL&INT Right
          - DEL&INT Left
          - Exit
        """
        # Reserve some space on the right side of the figure.
        # First, reserve space at the bottom of the figure for widgets.
        self.fig.subplots_adjust(bottom=0.1, right=0.8, left=0.25)  # Adjust as needed

        # For example, here we set an axes for check buttons in the right 15% of the figure.
        ax_check = self.fig.add_axes([0.82, 0.5, 0.15, 0.4], frameon=True)
        self.available_keypoints = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'all']
        # Initialize all keypoints as unselected.
        initial_states = [False] * len(self.available_keypoints)
        self.cb_keypoints = CheckButtons(ax_check, self.available_keypoints, initial_states)
        
        # Button for swapping left/right markers.
        ax_swap = self.fig.add_axes([0.82, 0.3, 0.15, 0.05])
        self.btn_swap_markers = Button(ax_swap, "Swap Markers")
        self.btn_swap_markers.on_clicked(self.on_swap_markers)
        
        # Buttons for DEL&INT operations.
        ax_delint_right = self.fig.add_axes([0.82, 0.25, 0.07, 0.05])
        self.btn_delint_right = Button(ax_delint_right, "DEL&INT R")
        self.btn_delint_right.on_clicked(self.on_delint_right)
        
        ax_delint_left = self.fig.add_axes([0.90, 0.25, 0.07, 0.05])
        self.btn_delint_left = Button(ax_delint_left, "DEL&INT L")
        self.btn_delint_left.on_clicked(self.on_delint_left)
        
        # Exit button (if desired).
        ax_exit = self.fig.add_axes([0.82, 0.08, 0.1, 0.05])
        self.btn_exit = Button(ax_exit, "Exit")
        self.btn_exit.on_clicked(self.on_exit)
        
        self.fig.canvas.draw()
    
    def _get_selected_keypoints(self):
        """
        Return a list of keypoints that are currently selected in the check buttons.
        The CheckButtons widget stores a list of booleans corresponding to each label.
        """
        # The CheckButtons widget (in recent Matplotlib versions) provides get_status()
        # which returns a list of booleans.
        status = self.cb_keypoints.get_status()  # list of booleans in the same order as self.available_keypoints
        selected = [kp for kp, active in zip(self.available_keypoints, status) if active]
        if not selected:
            print("No keypoints selected.")
        return selected

    def on_swap_markers(self, event):
        """
        For the current frame, swap left and right coordinate values for each selected keypoint.
        Records an undo action.
        """
        selected_kps = self._get_selected_keypoints()
        if not selected_kps:
            return
        
        mask = self.df['frame'] == self.current_frame
        
        if 'all' in selected_kps:
            selected_kps = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'elbow', 'wrist', 'ear','eye']

        # Build a list of column names that will be affected.
        cols_to_swap = []
        for kp in selected_kps:
            cols_to_swap.extend([f'left_{kp}_x', f'left_{kp}_y',
                                 f'right_{kp}_x', f'right_{kp}_y'])
        
        # Record the original values for undo.
        orig_values = self.df.loc[mask, cols_to_swap].copy()
        
        # Swap the values for each keypoint.
        for kp in selected_kps:
            lx = f'left_{kp}_x'
            ly = f'left_{kp}_y'
            rx = f'right_{kp}_x'
            ry = f'right_{kp}_y'
            
            # Swap x values.
            temp = self.df.loc[mask, lx].copy()
            self.df.loc[mask, lx] = self.df.loc[mask, rx]
            self.df.loc[mask, rx] = temp
            # Swap y values.
            temp = self.df.loc[mask, ly].copy()
            self.df.loc[mask, ly] = self.df.loc[mask, ry]
            self.df.loc[mask, ry] = temp
        
        def undo_swap():
            self.df.loc[mask, cols_to_swap] = orig_values
            self.plot_trajectories()
        
        self.record_action(undo_swap, f"Swapped markers for {selected_kps} in frame {self.current_frame}")
        print(f"Swapped markers for keypoints {selected_kps} in frame {self.current_frame}.")
        self.plot_trajectories()
        self.update_video_frame()
    
    def _interpolate_value(self, col, frame):
        """
        Given a column name and a frame number, perform a simple linear interpolation
        for the missing value at the specified frame using the nearest non-NaN values.
        Returns the interpolated value or NaN if interpolation is not possible.
        """
        # Get a subset of the data for the given column where the value is not NaN.
        df_col = self.df[['frame', col]].dropna().sort_values(by='frame')
        prev = df_col[df_col['frame'] < frame]
        nxt = df_col[df_col['frame'] > frame]
        if not prev.empty and not nxt.empty:
            row_prev = prev.iloc[-1]
            row_next = nxt.iloc[0]
            f_prev = row_prev['frame']
            f_next = row_next['frame']
            val_prev = row_prev[col]
            val_next = row_next[col]
            if f_next == f_prev:
                return val_prev
            # Linear interpolation.
            return val_prev + (val_next - val_prev) * (frame - f_prev) / (f_next - f_prev)
        else:
            return np.nan
    
    def on_delint_right(self, event):
        """
        For the current frame, for each selected keypoint, delete (set to NaN) the right-side
        coordinates and then replace them with an interpolated value.
        Records an undo action.
        """
        selected_kps = self._get_selected_keypoints()
        if not selected_kps:
            return
        
        if 'all' in selected_kps:
            selected_kps = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'elbow', 'wrist', 'ear','eye']

        mask = self.df['frame'] == self.current_frame
        # List of affected columns.
        affected_cols = []
        for kp in selected_kps:
            affected_cols.extend([f'right_{kp}_x', f'right_{kp}_y'])
        
        # Record original values for undo.
        orig_values = self.df.loc[mask, affected_cols].copy()
        
        # For each affected column, set current frame value to NaN and then interpolate.
        for col in affected_cols:
            # Delete (set to NaN).
            self.df.loc[mask, col] = np.nan
            # Compute the interpolated value.
            new_val = self._interpolate_value(col, self.current_frame)
            self.df.loc[mask, col] = new_val
        
        def undo_delint_right():
            self.df.loc[mask, affected_cols] = orig_values
            self.plot_trajectories()
        
        self.record_action(undo_delint_right,
                           f"DEL&INT Right for {selected_kps} in frame {self.current_frame}")
        print(f"Performed DEL&INT Right for keypoints {selected_kps} in frame {self.current_frame}.")
        self.plot_trajectories()
        self.update_video_frame()
    
    def on_delint_left(self, event):
        """
        For the current frame, for each selected keypoint, delete (set to NaN) the left-side
        coordinates and then replace them with an interpolated value.
        Records an undo action.
        """
        selected_kps = self._get_selected_keypoints()
        if not selected_kps:
            return
        
        if 'all' in selected_kps:
            selected_kps = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'elbow', 'wrist', 'ear','eye']
            
        mask = self.df['frame'] == self.current_frame
        affected_cols = []
        for kp in selected_kps:
            affected_cols.extend([f'left_{kp}_x', f'left_{kp}_y'])
        
        orig_values = self.df.loc[mask, affected_cols].copy()
        
        for col in affected_cols:
            self.df.loc[mask, col] = np.nan
            new_val = self._interpolate_value(col, self.current_frame)
            self.df.loc[mask, col] = new_val
        
        def undo_delint_left():
            self.df.loc[mask, affected_cols] = orig_values
            self.plot_trajectories()
        
        self.record_action(undo_delint_left,
                           f"DEL&INT Left for {selected_kps} in frame {self.current_frame}")
        print(f"Performed DEL&INT Left for keypoints {selected_kps} in frame {self.current_frame}.")
        self.plot_trajectories()
        self.update_video_frame()
    
    def on_exit(self, event):
        """
        Exit the tool by closing the Matplotlib figure.
        """
        print("Exiting OutlierCleaningTool.")
        plt.close(self.fig)
