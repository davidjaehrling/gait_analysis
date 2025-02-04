import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import time


class BaseInteractiveTool:
    """
    A parent class that provides:
      - A Matplotlib window displaying a trajectory plot (frame number on x-axis and
        a selectable coordinate on the y-axis).
      - A CV2 window that shows the video frame with skeletons drawn for all persons.
      - Navigation of frames by clicking on the trajectory plot and by using the
        left/right arrow keys.
      - A built-in undo mechanism for reverting the last widget-driven action.
      
    Child classes can extend the widget area (e.g. add text boxes or extra buttons)
    to modify the underlying DataFrame or perform more specialized actions.
    """

    def __init__(self, df: pd.DataFrame, video_path: str, skeleton_drawer,
                 y_keys=None, default_y_key=None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame must contain at least the columns ['frame', 'person_idx'] and any
            additional coordinate columns that you want to plot (e.g. 'right_ankle_x').
        video_path : str
            Path to the associated video file.
        skeleton_drawer : object
            An object with a method like draw_skeleton(frame, df_frame, person_idx)
            that renders the skeleton on a given video frame.
        y_keys : list of str, optional
            List of candidate column names in df to be used as y-axis data in the
            trajectory plot. If None, then only default_y_key (if provided) will be used.
        default_y_key : str, optional
            The default column name to plot on the y-axis. If not provided and y_keys
            is given, the first element of y_keys is used.
        """
        # Store a copy of the DataFrame
        self.df = df.copy()
        self.video_path = video_path
        self.skeleton_drawer = skeleton_drawer

        # Ensure the DataFrame has required columns.
        if 'frame' not in self.df.columns or 'person_idx' not in self.df.columns:
            raise ValueError("DataFrame must contain at least 'frame' and 'person_idx' columns.")

        # Sort the DataFrame by frame (and then person_idx) for consistency.
        self.df.sort_values(by=['frame', 'person_idx'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Setup coordinate options for the y-axis in the trajectory plot.
        if y_keys is None:
            # If no list is provided, use default_y_key if available.
            self.y_keys = [default_y_key] if default_y_key is not None else []
        else:
            self.y_keys = y_keys

        if default_y_key is None and self.y_keys:
            self.current_y_key = self.y_keys[0]
        else:
            self.current_y_key = default_y_key

        # Initialize the current frame (starting at 0)
        self.current_frame = 0

        # History of widget actions for undo functionality.
        # Each entry is a tuple: (undo_function, description)
        self.history = []

        # Open the video capture.
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Warning: could not open video: {self.video_path}")

        # Create the Matplotlib figure and axis for the trajectory plot.
        self.fig = plt.figure(figsize=(10, 6))
        self.ax_main = self.fig.add_subplot(111)
        self.plot_trajectories()

        # Initialize default widgets.
        self._init_widgets()

        # Connect Matplotlib events (e.g. for clicking on the plot).
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Enable interactive mode.
        plt.ion()

    # -----------------------------
    # 1. Plotting and Widgets Setup
    # -----------------------------
    def plot_trajectories(self):
        """
        Plot each person's trajectory using the current_y_key column (if available)
        against the frame number.
        """
        self.ax_main.clear()
        if self.current_y_key not in self.df.columns:
            self.ax_main.text(0.5, 0.5, f"Column '{self.current_y_key}' not found in DataFrame.",
                              transform=self.ax_main.transAxes,
                              ha="center", va="center")
        else:
            unique_pids = sorted(self.df['person_idx'].unique())
            for pid in unique_pids:
                sub_df = self.df[self.df['person_idx'] == pid]
                self.ax_main.plot(sub_df['frame'], sub_df[self.current_y_key],
                                  label=f"PID {pid}")
            self.ax_main.set_xlabel("Frame")
            self.ax_main.set_ylabel(self.current_y_key)
            self.ax_main.set_title("Trajectory Plot (click to jump to a frame)")
            self.ax_main.legend()

        self.fig.canvas.draw()

    def _init_widgets(self):
        """
        Initialize default widgets in the Matplotlib figure:
         - A RadioButtons widget to select the y-axis coordinate (if more than one option).
         - An Undo button to revert the last action.
         
        Child classes can extend this method to add more buttons or text boxes.
        """
        # Example: Create a RadioButtons widget if more than one y_key is provided.
        if len(self.y_keys) > 1:
            ax_radio = self.fig.add_axes([0.02, 0.5, 0.15, 0.3], frameon=True)
            self.radio_y = RadioButtons(ax_radio, self.y_keys,
                                         active=self.y_keys.index(self.current_y_key))
            self.radio_y.on_clicked(self.on_y_key_change)
        else:
            self.radio_y = None

        # Create an Undo button.
        ax_undo = self.fig.add_axes([0.8, 0.02, 0.1, 0.05])
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(self.on_undo)

        # (Child classes may add more widget buttons and text boxes as needed.)

    def on_y_key_change(self, label):
        """
        Callback for when the user selects a different y-axis coordinate.
        """
        self.current_y_key = label
        print(f"Y-axis coordinate changed to: {self.current_y_key}")
        self.plot_trajectories()

    # -----------------------------
    # 2. Video Display and Navigation
    # -----------------------------
    def update_video_frame(self):
        """
        Seek to the current frame in the video, draw skeleton overlays, and
        display the frame (with the current frame number drawn) in a CV2 window.
        """
        if not self.cap.isOpened():
            print("Video capture is not open.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Could not read frame {self.current_frame}.")
            return

        # Get all rows in the DataFrame corresponding to the current frame.
        df_frame = self.df[self.df['frame'] == self.current_frame]

        # Draw the skeleton for each person in this frame.
        for person_idx in df_frame['person_idx'].unique():
            self.skeleton_drawer.draw_skeleton(frame, df_frame, person_idx)

        # Overlay the current frame number on the image.
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Reduce resolution for display (optional).
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
        # Show the frame in a CV2 window.
        cv2.imshow("Video", frame)

    def on_click(self, event):
        """
        Matplotlib callback: when the user clicks on the trajectory plot,
        jump to the corresponding frame.
        """
        if event.inaxes != self.ax_main:
            return
        if event.xdata is None:
            return

        frame_clicked = int(round(event.xdata))
        self.goto_frame(frame_clicked)

    def goto_frame(self, frame_number):
        """
        Set the current frame and update the CV2 window.
        """
        self.current_frame = frame_number
        print(f"Jumping to frame {self.current_frame}")
        self.update_video_frame()

    def goto_next_frame(self):
        """Go to the next frame."""
        self.goto_frame(self.current_frame + 1)

    def goto_previous_frame(self):
        """Go to the previous frame (if not at frame 0)."""
        if self.current_frame > 0:
            self.goto_frame(self.current_frame - 1)

    # -----------------------------
    # 3. Undo Mechanism
    # -----------------------------
    def record_action(self, undo_fn, description=""):
        """
        Record an action (e.g., a DataFrame change) with an associated undo function.
        When undo is triggered, the undo_fn is called.

        Parameters
        ----------
        undo_fn : callable
            A function with no arguments that reverts the recorded action.
        description : str, optional
            A short description of the action (used for logging).
        """
        self.history.append((undo_fn, description))
        print(f"Action recorded: {description}")

    def undo_last_action(self, event=None):
        """
        Undo the last recorded action.
        """
        if not self.history:
            print("No actions to undo.")
            return
        undo_fn, description = self.history.pop()
        print(f"Undoing action: {description}")
        undo_fn()
        # After undoing, you might want to replot the trajectories.
        self.plot_trajectories()

    def on_undo(self, event):
        """
        Callback for the Undo button.
        """
        self.undo_last_action()

    # -----------------------------
    # 4. Main Run Loop
    # -----------------------------
    def run(self):
        """
        Start the interactive tool. This method enters a loop that:
          - Keeps updating the CV2 window.
          - Checks for left/right arrow key presses to move forward/backward by one frame.
          - Exits when the Matplotlib figure is closed (or ESC is pressed in the CV2 window).

        The child classes can call this method to run the complete tool.
        """
        print("Starting BaseInteractiveTool. Close the Matplotlib window to exit.")
        # Display the initial frame.
        self.update_video_frame()

        # Main loop.
        while plt.fignum_exists(self.fig.number):
            # cv2.waitKey returns -1 if no key is pressed.
            key = cv2.waitKey(30) & 0xFF

            # Some systems return 255 when no key is pressed.
            if key != 255:
                # Note: The key codes for arrow keys vary by system.
                # Commonly, 81 is the left arrow and 83 is the right arrow.
                if key == 98:
                    self.goto_previous_frame()
                elif key == 110:
                    self.goto_next_frame()
                elif key == 27:  # ESC key pressed: exit.
                    print("ESC pressed. Exiting.")
                    plt.close(self.fig)
                    break

            # Allow the Matplotlib event loop to process events.
            plt.pause(0.001)
            # (Optional) Sleep a short time to avoid a busy loop.
            time.sleep(0.01)

        # Cleanup: close the CV2 window and release the video capture.
        cv2.destroyAllWindows()
        if self.cap.isOpened():
            self.cap.release()
        print("Exiting BaseInteractiveTool.")
        return self.df
