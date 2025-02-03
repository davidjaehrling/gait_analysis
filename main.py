from data_io.openpose_reader import OpenPoseReader
from data_io.alphapose_reader import AlphaPoseReader
from preprocessing.tracker import PersonTracker
from preprocessing.outlier_detection import OutlierDetector
from preprocessing.interpolation import Interpolator
from analysis.angle_calculator import AngleCalculator
from manual_correction.reindexing_tool import ReindexingTool
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

            trials_path = os.path.join(path_keypoints, "csv", model_name, participant)
            trials = os.listdir(trials_path)
            trials = trials[2:3]
            for trial in trials:
                    
                # 1) Load data
                keypoint_path = os.path.join(path_keypoints, "csv", model_name, participant, trial)
                df_raw = reader.load_csv(keypoint_path)
                

                # 2) Track persons
                tracker = PersonTracker(max_distance=100, max_age=100, velocity_history=30)
                df_tracked = tracker.track(df_raw)

                video_path = os.path.join(path_videos, participant, "Cut", f"{trial[:-4]}.mp4")
                visualizer = SkeletonVisualizer(skeleton_definition=skeleton, keypoints_definition=keypoint_dict)
                #visualizer.visualize_video(video_path, df_tracked)


                # 3) Manual tracking rerassignment
                tool = ReindexingTool(df_tracked, video_path, visualizer, 
                                    x_key='right_ankle_x', 
                                    y_key='right_ankle_y')

                df_corrected = tool.run()

                # delete all non person_idx == 0 rows
                df_corrected = df_corrected[df_corrected["person_idx"] == 0]

                #visualizer.visualize_video(video_path, df_corrected)

                # 4) Outlier detection
                print("peinis")
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