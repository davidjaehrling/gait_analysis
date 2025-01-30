import cv2
import pandas as pd

class ReindexingTool:
    """
    GUI-based or CLI-based tool that lets you visualize frames and 
    manually assign the correct 'person_idx' if automatic tracking fails.
    """

    def __init__(self):
        pass

    def run(self, video_path: str, df: pd.DataFrame):
        """
        - Open a window using OpenCV.
        - Show skeleton overlays for each frame or every Nth frame.
        - If user notices a mislabeled person, allow them to press a key 
          to change the label or confirm the correct ID.
        - Return the corrected DataFrame.
        """
        # Implementation depends on how interactive you want it.
        return df
