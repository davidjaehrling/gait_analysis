from data_io.openpose_reader import OpenPoseReader
from data_io.alphapose_reader import AlphaPoseReader
from preprocessing.tracker import PersonTracker
from preprocessing.outlier_detection import OutlierDetector
from preprocessing.interpolation import Interpolator
from analysis.angle_calculator import AngleCalculator
from visualization.skeleton_visualizer import SkeletonVisualizer
from utils.paths import path_videos, path_keypoints

import numpy as np
import os


def main():
    model_readers = {
        "openpose": OpenPoseReader(),
        #"alphapose": AlphaPoseReader(),
    }
    
    angle_definitions = {
        "knee_angle": ("3pt", ("right_hip", "right_knee", "right_ankle")),
        # ...
    }
    angle_calculator = AngleCalculator(angle_definitions)

    participants = [i for i in os.listdir(path_videos) if i.startswith("P")]


    for model_name, reader in model_readers.items():
        print(f"Processing: {model_name}")
        if model_name == "openpose":
            from config.keypoint_dict import openpose_keypoints as keypoint_dict
            from config.keypoint_dict import openpose_skeleton as skeleton
        elif model_name == "alphapose":
            from config.keypoint_dict import alphapose_keypoints as keypoint_dict
            from config.keypoint_dict import alphapose_skeleton as skeleton
        for participant in participants:

            trials = os.listdir(f"{path_keypoints}\csv\{model_name}\{participant}")
            for trial in trials:
                    
                # 1) Load data
                df_raw = reader.load_csv(f"{path_keypoints}\csv\{model_name}\{participant}\{trial}")
                

                # 2) Track persons
                tracker = PersonTracker(max_distance=50, max_age=100, velocity_history=100)
                df_tracked = tracker.track(df_raw)

                visualizer = SkeletonVisualizer(skeleton_definition=skeleton, keypoints_definition=keypoint_dict)
                visualizer.visualize_video(f"{path_videos}\{participant}\Cut\{trial[:-4]}.mp4", df_tracked)


                # 3) Manual tracking rerassignment


                '''
                outlier_detector = OutlierDetector()
                # build your joint_pairs ...
                outliers = outlier_detector.combine_outliers(df_tracked, joint_pairs=...)

                # Set outliers to NaN
                for xcol, ycol in ...:
                    df_tracked.loc[outliers, xcol] = np.nan
                    df_tracked.loc[outliers, ycol] = np.nan

                # 4) Interpolate
                interpolator = Interpolator(method='linear')
                df_cleaned = interpolator.interpolate(df_tracked, columns_of_interest)

                # 5) Angle calculation
                df_angles = angle_calculator.compute_angles(df_cleaned)

                # 6) Save final
                df_angles.to_csv(f"/path/to/output/{model_name}_angles.csv", index=False)

                # 7) Possibly visualize or do manual corrections in between
                # ...
                '''
if __name__ == "__main__":
    main()  