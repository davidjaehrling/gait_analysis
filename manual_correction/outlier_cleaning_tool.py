import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from manual_correction.base_interactive_tool import BaseInteractiveTool

class OutlierCleaningTool(BaseInteractiveTool):
    """
    A child class of BaseInteractiveTool for cleaning outliers.
    
    Provides the following functionalities:
      - Swap left/right markers for the currently selected frame.
        The user selects one or more keypoints (e.g. ankle, knee, hip, shoulder,
        heel, big_toe) via check buttons (displayed on the right side of the figure)
        and clicks the "Swap Markers" button.
      - Delete (set to NaN) the coordinates for one side (right or left) for the
        selected keypoints in the current frame. Two separate buttons ("DEL R" and "DEL L")
        perform deletion.
      - An additional "Interpolate" button that, when pressed, interpolates over all 
        coordinate columns (i.e. those ending in '_x' or '_y') for all frames where the values are NaN.
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
        We reserve a portion of the figure on the right for a set of check buttons (to select keypoints)
        and add buttons for:
          - Swap Markers
          - DEL R (delete right-side coordinates in the current frame)
          - DEL L (delete left-side coordinates in the current frame)
          - Interpolate (apply interpolation over all frames to fill NaN values)
          - Exit
        """
        # Reserve space at the bottom and right of the figure for widgets.
        self.fig.subplots_adjust(bottom=0.1, right=0.8, left=0.25)  # Adjust as needed

        # CheckButtons for keypoint selection.
        ax_check = self.fig.add_axes([0.82, 0.5, 0.15, 0.4], frameon=True)
        self.available_keypoints = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'all']
        initial_states = [False] * len(self.available_keypoints)
        self.cb_keypoints = CheckButtons(ax_check, self.available_keypoints, initial_states)
        
        # Button for swapping left/right markers.
        ax_swap = self.fig.add_axes([0.82, 0.3, 0.15, 0.05])
        self.btn_swap_markers = Button(ax_swap, "Swap Markers")
        self.btn_swap_markers.on_clicked(self.on_swap_markers)
        
        # Buttons for deletion only (set selected coordinate values to NaN).
        ax_del_right = self.fig.add_axes([0.82, 0.25, 0.07, 0.05])
        self.btn_del_right = Button(ax_del_right, "DEL R")
        self.btn_del_right.on_clicked(self.on_del_right)
        
        ax_del_left = self.fig.add_axes([0.90, 0.25, 0.07, 0.05])
        self.btn_del_left = Button(ax_del_left, "DEL L")
        self.btn_del_left.on_clicked(self.on_del_left)
        
        # Button for interpolating all NaN values (both left and right) in the entire DataFrame.
        ax_interp = self.fig.add_axes([0.82, 0.18, 0.15, 0.05])
        self.btn_interp = Button(ax_interp, "Interpolate")
        self.btn_interp.on_clicked(self.on_interpolate_all)
        
        # Exit button.
        ax_exit = self.fig.add_axes([0.82, 0.08, 0.1, 0.05])
        self.btn_exit = Button(ax_exit, "Exit")
        self.btn_exit.on_clicked(self.on_exit)
        
        self.fig.canvas.draw()
    
    def _get_selected_keypoints(self):
        """
        Return a list of keypoints that are currently selected in the check buttons.
        """
        status = self.cb_keypoints.get_status()
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
            selected_kps = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'elbow', 'wrist', 'ear', 'eye']

        cols_to_swap = []
        for kp in selected_kps:
            cols_to_swap.extend([f'left_{kp}_x', f'left_{kp}_y',
                                 f'right_{kp}_x', f'right_{kp}_y'])
        
        orig_values = self.df.loc[mask, cols_to_swap].copy()
        
        for kp in selected_kps:
            lx = f'left_{kp}_x'
            ly = f'left_{kp}_y'
            rx = f'right_{kp}_x'
            ry = f'right_{kp}_y'
            
            temp = self.df.loc[mask, lx].copy()
            self.df.loc[mask, lx] = self.df.loc[mask, rx]
            self.df.loc[mask, rx] = temp
            
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
            return val_prev + (val_next - val_prev) * (frame - f_prev) / (f_next - f_prev)
        else:
            return np.nan
    
    def on_del_right(self, event):
        """
        For the current frame, for each selected keypoint, set the right-side coordinate values to NaN.
        Records an undo action.
        """
        selected_kps = self._get_selected_keypoints()
        if not selected_kps:
            return
        
        if 'all' in selected_kps:
            selected_kps = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'elbow', 'wrist', 'ear', 'eye']
        
        mask = self.df['frame'] == self.current_frame
        affected_cols = []
        for kp in selected_kps:
            affected_cols.extend([f'right_{kp}_x', f'right_{kp}_y'])
        
        orig_values = self.df.loc[mask, affected_cols].copy()
        for col in affected_cols:
            self.df.loc[mask, col] = np.nan
        
        def undo_del_right():
            self.df.loc[mask, affected_cols] = orig_values
            self.plot_trajectories()
        self.record_action(undo_del_right, f"Deleted right keypoints {selected_kps} in frame {self.current_frame}")
        print(f"Deleted right keypoints {selected_kps} in frame {self.current_frame}.")
        self.plot_trajectories()
        self.update_video_frame()
    
    def on_del_left(self, event):
        """
        For the current frame, for each selected keypoint, set the left-side coordinate values to NaN.
        Records an undo action.
        """
        selected_kps = self._get_selected_keypoints()
        if not selected_kps:
            return
        
        if 'all' in selected_kps:
            selected_kps = ['shoulder', 'hip', 'knee', 'ankle', 'heel', 'big_toe', 'small_toe', 'elbow', 'wrist', 'ear', 'eye']
            
        mask = self.df['frame'] == self.current_frame
        affected_cols = []
        for kp in selected_kps:
            affected_cols.extend([f'left_{kp}_x', f'left_{kp}_y'])
        
        orig_values = self.df.loc[mask, affected_cols].copy()
        for col in affected_cols:
            self.df.loc[mask, col] = np.nan
        
        def undo_del_left():
            self.df.loc[mask, affected_cols] = orig_values
            self.plot_trajectories()
        self.record_action(undo_del_left, f"Deleted left keypoints {selected_kps} in frame {self.current_frame}")
        print(f"Deleted left keypoints {selected_kps} in frame {self.current_frame}.")
        self.plot_trajectories()
        self.update_video_frame()
    
    def on_interpolate_all(self, event):
        """
        Interpolates all coordinate columns (those ending in '_x' or '_y') in the entire DataFrame.
        This will fill in all NaN values using linear interpolation (forward/backward).
        Records an undo action.
        """
        # Identify coordinate columns.
        coordinate_cols = [col for col in self.df.columns if col.endswith('_x') or col.endswith('_y')]
        # Record a copy of the current values for undo.
        orig_values = self.df[coordinate_cols].copy()
        
        # Interpolate over the entire DataFrame.
        self.df[coordinate_cols] = self.df[coordinate_cols].interpolate(method='linear', limit_direction='both')
        self.df[coordinate_cols] = self.df[coordinate_cols].bfill().ffill()
        print("Performed interpolation on all coordinate columns over the entire DataFrame.")
        
        def undo_interp():
            self.df[coordinate_cols] = orig_values
            self.plot_trajectories()
        self.record_action(undo_interp, "Interpolated all coordinate columns")
        
        self.plot_trajectories()
        self.update_video_frame()
    
    def on_exit(self, event):
        """
        Exit the tool by closing the Matplotlib figure.
        """
        print("Exiting OutlierCleaningTool.")
        plt.close(self.fig)
