import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from manual_correction.base_interactive_tool import BaseInteractiveTool



class ReindexingTool(BaseInteractiveTool):
    """
    A child class of BaseInteractiveTool that adds widgets for:
      - Deleting a given person_idx completely from the data.
      - Swapping a given person_idx with person_idx==0 in a specified frame range.
      - Exiting the tool.
      
    Both deletion and swap operations are recorded so that the Undo button (inherited
    from BaseInteractiveTool) can revert the last action.
    """
    def __init__(self, df: pd.DataFrame, video_path: str, skeleton_drawer,
                 y_keys=None, default_y_key=None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The keypoint data. Must contain at least the columns 'frame' and 'person_idx',
            as well as any coordinate columns used for plotting.
        video_path : str
            Path to the video associated with the data.
        skeleton_drawer : object
            An object with a method `draw_skeleton(frame, df_frame, person_idx)`
            that renders the keypoint skeleton on a video frame.
        y_keys : list of str, optional
            List of possible coordinate columns to use for the y-axis in the trajectory plot.
        default_y_key : str, optional
            The default coordinate to plot on the y-axis.
        """
        super().__init__(df, video_path, skeleton_drawer, y_keys, default_y_key)
        self._init_extra_widgets()

    def _init_extra_widgets(self):
        """
        Add extra widgets to the Matplotlib figure beneath the plot.
        """
        # First, reserve space at the bottom of the figure for widgets.
        self.fig.subplots_adjust(bottom=0.25, left=0.25)  # Adjust as needed

        # --- Deletion Widget ---
        # Create a TextBox to enter the person_idx to delete.
        ax_del = self.fig.add_axes([0.4, 0.03, 0.1, 0.05])  # Now placed near the bottom
        self.tb_del = TextBox(ax_del, "Del PID", initial="")
        # Create a button to trigger deletion.
        ax_del_btn = self.fig.add_axes([0.55, 0.03, 0.1, 0.05])
        self.btn_del = Button(ax_del_btn, "Delete")
        self.btn_del.on_clicked(self.on_delete)

        # --- Swap Widget (swap given PID with 0) ---
        # TextBox for the person_idx to swap with 0.
        ax_swap_pid = self.fig.add_axes([0.1, 0.11, 0.1, 0.05])
        self.tb_swap_pid = TextBox(ax_swap_pid, "Swap PID", initial="")
        # TextBox for start frame.
        ax_swap_start = self.fig.add_axes([0.25, 0.11, 0.1, 0.05])
        self.tb_swap_start = TextBox(ax_swap_start, "Start", initial="")
        # TextBox for end frame.
        ax_swap_end = self.fig.add_axes([0.4, 0.11, 0.1, 0.05])
        self.tb_swap_end = TextBox(ax_swap_end, "End", initial="")
        # Button to trigger the swap.
        ax_swap_btn = self.fig.add_axes([0.55, 0.11, 0.1, 0.05])
        self.btn_swap = Button(ax_swap_btn, "Swap with 0")
        self.btn_swap.on_clicked(self.on_swap)

        # --- Exit Widget ---
        # A button to exit the tool.
        ax_exit = self.fig.add_axes([0.8, 0.1, 0.1, 0.05])
        self.btn_exit = Button(ax_exit, "Exit")
        self.btn_exit.on_clicked(self.on_exit)

        self.fig.canvas.draw()


    # --- Callback for Deletion ---
    def on_delete(self, event):
        """
        Delete all keypoint data (rows) corresponding to the person_idx entered in the deletion TextBox.
        Records an undo action so the deletion can be reverted.
        """
        try:
            pid = int(self.tb_del.text)
        except ValueError:
            print("Invalid PID value for deletion.")
            return

        mask = (self.df['person_idx'] == pid)
        if mask.sum() == 0:
            print(f"No entries found for person_idx {pid}.")
            return

        # Save the rows to be deleted so they can be reinserted on undo.
        deleted_rows = self.df[mask].copy()

        def undo_delete():
            # Reinsert the deleted rows and re-sort the DataFrame.
            self.df = pd.concat([self.df, deleted_rows], ignore_index=True)
            self.df.sort_values(by=['frame', 'person_idx'], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.plot_trajectories()

        self.record_action(undo_delete, f"Delete person_idx {pid}")

        # Delete the rows.
        self.df = self.df[~mask].copy()
        self.df.reset_index(drop=True, inplace=True)
        print(f"Deleted person_idx {pid}.")
        self.plot_trajectories()

    # --- Callback for Swapping ---
    def on_swap(self, event):
        """
        Swap the person_idx provided in the swap TextBox with person_idx == 0
        over the frame range specified by the Start and End TextBoxes.
        Records an undo action so the swap can be reverted.
        """
        try:
            pid = int(self.tb_swap_pid.text)
            start_frame = int(self.tb_swap_start.text)
            end_frame = int(self.tb_swap_end.text)
        except ValueError:
            print("Invalid input for swap operation.")
            return

        # Ensure proper frame ordering.
        f1 = min(start_frame, end_frame)
        f2 = max(start_frame, end_frame)
        # Find all rows in the specified frame range where person_idx is either pid or 0.
        mask = ((self.df['frame'] >= f1) & (self.df['frame'] <= f2) &
                (self.df['person_idx'].isin([pid, 0])))

        if mask.sum() == 0:
            print("No entries found for the swap operation in the specified frame range.")
            return

        # Record the original person_idx values for these rows.
        previous_values = self.df.loc[mask, 'person_idx'].copy()

        def undo_swap():
            self.df.loc[mask, 'person_idx'] = previous_values
            self.plot_trajectories()

        self.record_action(undo_swap, f"Swap person_idx {pid} with 0 in frames [{f1}, {f2}]")

        # Perform the swap using a temporary placeholder.
        temp_val = -9999  # temporary value (assumed not to occur in the data)
        self.df.loc[mask & (self.df['person_idx'] == pid), 'person_idx'] = temp_val
        self.df.loc[mask & (self.df['person_idx'] == 0), 'person_idx'] = pid
        self.df.loc[mask & (self.df['person_idx'] == temp_val), 'person_idx'] = 0

        print(f"Swapped person_idx {pid} with 0 in frames [{f1}, {f2}].")
        self.plot_trajectories()

    # --- Callback for Exiting ---
    def on_exit(self, event):
        """
        Exit the ReindexingTool. This closes the Matplotlib figure
        and thereby ends the interactive session.
        """
        print("Exiting ReindexingTool.")
        print(self.df)
        plt.close(self.fig)
