import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import TextBox, Button

class ReindexingTool:
    """
    A GUI tool (using Matplotlib + OpenCV) to manually correct person_idx 
    over a range of frames by visually inspecting a chosen keypoint trajectory
    and the corresponding video frames with skeleton overlay.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 video_path: str, 
                 skeleton_drawer,
                 x_key='right_ankle_x', 
                 y_key='right_ankle_y'):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must include columns: ['frame', 'person_idx', x_key, y_key, ...].
        video_path : str
            Path to the video associated with df.
        skeleton_drawer : object
            Your visualization object with a method like 
            `draw_skeleton(frame, df_for_that_frame)` 
            or similar for rendering.
        x_key, y_key : str
            The keypoint columns to plot on the trajectory (e.g., right_ankle_x / right_ankle_y).
            We'll only plot x_key vs. 'frame' for each person_idx, but you could adapt it.
        """
        # Make a copy so we don't mutate original DataFrame
        self.df = df.copy()
        self.video_path = video_path
        self.skeleton_drawer = skeleton_drawer
        self.x_key = x_key
        self.y_key = y_key

        # Ensure required columns exist
        required_cols = {'frame', 'person_idx', self.x_key, self.y_key}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"DataFrame is missing one or more required columns: {required_cols}")

        # Sort by frame to keep things consistent
        self.df.sort_values(by=['frame', 'person_idx'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Identify unique person_idx values
        self.unique_pids = sorted(self.df['person_idx'].unique())

        # State variables for swapping
        self.swap_pid1 = None
        self.swap_pid2 = None
        self.swap_start_frame = None
        self.swap_end_frame = None

        # Video Capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Warning: could not open video: {self.video_path}")

        # Set up the Matplotlib figure and main trajectory axis
        # We will add widgets (TextBoxes, Buttons) below the plot
        self.fig = plt.figure(figsize=(9, 6))
        self.ax_main = self.fig.add_subplot(111)

        # Prepare the main trajectory plot
        self._plot_trajectories()

        # Turn on interactive mode
        plt.ion()

        # Connect the mouse click event on the main axis
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Create widget areas (axes) in the figure for textboxes and buttons
        self._init_widgets()

    # -------------------------------------------------------------------------
    # 1. Plotting
    # -------------------------------------------------------------------------
    def _plot_trajectories(self):
        """
        Draw each person's x_key vs. frame trajectory as a line in the main axes.
        """
        self.ax_main.clear()
        for pid in self.unique_pids:
            sub_df = self.df[self.df['person_idx'] == pid]
            self.ax_main.plot(sub_df['frame'], sub_df[self.x_key], label=f"PID {pid}")

        self.ax_main.set_xlabel("Frame")
        self.ax_main.set_ylabel(self.x_key)
        self.ax_main.set_title("Click on the trajectory to jump to that frame.")
        self.ax_main.legend()
        self.fig.canvas.draw()

    # -------------------------------------------------------------------------
    # 2. Matplotlib Widgets: TextBoxes and Buttons
    # -------------------------------------------------------------------------
    def _init_widgets(self):
        """
        Create small text boxes and a button to swap person indices in a given frame range.
        We'll place them in the lower region of the figure using the add_axes approach.
        """
        # coords: [left, bottom, width, height] in figure fraction
        ax_pid1 = self.fig.add_axes([0.1, 0.02, 0.08, 0.05])
        ax_pid2 = self.fig.add_axes([0.23, 0.02, 0.08, 0.05])
        ax_start = self.fig.add_axes([0.36, 0.02, 0.08, 0.05])
        ax_end   = self.fig.add_axes([0.49, 0.02, 0.08, 0.05])

        # Create textboxes
        self.tb_pid1 = TextBox(ax_pid1, 'PID1', initial="0")
        self.tb_pid2 = TextBox(ax_pid2, 'PID2', initial="1")
        self.tb_start = TextBox(ax_start, 'Start', initial="0")
        self.tb_end = TextBox(ax_end, 'End', initial="100")

        # Button
        ax_button_swap = self.fig.add_axes([0.62, 0.02, 0.08, 0.05])
        self.btn_swap = Button(ax_button_swap, 'Swap')
        self.btn_swap.on_clicked(self.on_swap_button)

        # Optionally, add a "Quit" button to close
        ax_button_quit = self.fig.add_axes([0.75, 0.02, 0.08, 0.05])
        self.btn_quit = Button(ax_button_quit, 'Quit')
        self.btn_quit.on_clicked(self.on_quit_button)

    # -------------------------------------------------------------------------
    # 3. Event Callbacks
    # -------------------------------------------------------------------------
    def on_click(self, event):
        """
        Matplotlib callback for a mouse click in the figure.
        We only care if the click is in data coords on the main axis.
        """
        # Check if click was in the main axis
        if event.inaxes != self.ax_main:
            return

        # The xdata is the frame number
        if event.xdata is None:
            return

        frame_clicked = int(round(event.xdata))
        self.show_frame(frame_clicked)

    def on_swap_button(self, event):
        """
        Callback when user presses the "Swap" button.
        Reads the text boxes for PID1, PID2, start, end frames 
        and calls do_swap().
        """
        try:
            pid1 = int(self.tb_pid1.text)
            pid2 = int(self.tb_pid2.text)
            fstart = int(self.tb_start.text)
            fend = int(self.tb_end.text)
        except ValueError:
            print("Invalid integer in text boxes. Swap aborted.")
            return

        self.do_swap(pid1, pid2, fstart, fend)

    def on_quit_button(self, event):
        """
        Closes the Matplotlib figure and ends the interactive session.
        """
        plt.close(self.fig)  # This will unblock plt.show()

    # -------------------------------------------------------------------------
    # 4. Video Display with Skeleton Overlay
    # -------------------------------------------------------------------------
    def show_frame(self, frame_number):
        """
        Jump to the given frame_number in the video, 
        draw the skeleton overlay, and show in an OpenCV window.
        """
        if not self.cap.isOpened():
            print("Video is not opened.")
            return

        # Seek to the frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Could not read frame {frame_number}.")
            return

        # Filter df to rows for the current frame
        df_frame = self.df[self.df['frame'] == frame_number]

        # Use skeleton_drawer to draw
        # The exact method name depends on your skeleton_drawer implementation
        # For example:
        for person_idx in df_frame["person_idx"].unique():
                self.skeleton_drawer.draw_skeleton(frame, df_frame, person_idx)
      
        # Resize the frame for display
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        
        # Show in a window
        cv2.imshow("Reindexing Tool - Skeleton", frame)
        cv2.waitKey(1)  # small delay so window refreshes

    # -------------------------------------------------------------------------
    # 5. Swapping Indices
    # -------------------------------------------------------------------------
    def do_swap(self, pid1, pid2, start_frame, end_frame):
        """
        Swap two person indices (pid1, pid2) for frames in [start_frame, end_frame].
        """
        # Make sure start <= end
        f1 = min(start_frame, end_frame)
        f2 = max(start_frame, end_frame)

        print(f"Swapping PID {pid1} <-> PID {pid2} in frames [{f1}, {f2}]...")

        # Condition: frames in [f1, f2] and person_idx is either pid1 or pid2
        mask = (self.df['frame'] >= f1) & (self.df['frame'] <= f2) & (self.df['person_idx'].isin([pid1, pid2]))

        # We'll do a standard 3-step swap
        temp_val = 99999  # hopefully not used in your data
        self.df.loc[mask & (self.df['person_idx'] == pid1), 'person_idx'] = temp_val
        self.df.loc[mask & (self.df['person_idx'] == pid2), 'person_idx'] = pid1
        self.df.loc[mask & (self.df['person_idx'] == temp_val), 'person_idx'] = pid2

        # Re-plot to reflect changes
        self._plot_trajectories()

    # -------------------------------------------------------------------------
    # 6. Main Runner
    # -------------------------------------------------------------------------
    def run(self):
        """
        Start the interactive session. 
        Blocks until the user closes the plot window.
        Returns the updated DataFrame with final, user-corrected person_idx.
        """
        print("Reindexing Tool is running. Close the plot window to finish.")
        # This will block until the figure is closed.
        plt.show(block=True)

        # Once figure is closed, also destroy the OpenCV window
        cv2.destroyAllWindows()
        self.cap.release()

        print("Reindexing Tool finished. Returning corrected DataFrame.")
        return self.df
