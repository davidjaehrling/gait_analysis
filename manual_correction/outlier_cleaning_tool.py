class OutlierCleaningTool:
    """
    Similar approach: let user see a time-series plot or 
    skeleton overlay, highlight suspected outliers, 
    and confirm or remove them interactively.
    """

    def __init__(self):
        pass

    def run(self, video_path: str, df: pd.DataFrame):
        """
        - Possibly plot the trajectory of one keypoint in Matplotlib
          or overlay in OpenCV.
        - Wait for user to remove or keep outliers, or do an interpolation on selection.
        """
        return df
