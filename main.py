from data_io.openpose_reader import OpenPoseReader
from data_io.alphapose_reader import AlphaPoseReader
from preprocessing.tracker import PersonTracker
from preprocessing.outlier_detection import OutlierDetector
from preprocessing.interpolation import Interpolator
from analysis.angle_calculator import AngleCalculator
from manual_correction.reindexing_tool import ReindexingTool
from manual_correction.outlier_cleaning_tool import OutlierCleaningTool
from visualization.skeleton_visualizer import SkeletonVisualizer
from utils.paths import path_videos, path_keypoints
from utils.tdto import get_first_event, distinct_angles

import numpy as np
import os
import shutil

def main():
    model_readers = {
        "openpose": OpenPoseReader(),
        #"alphapose": AlphaPoseReader(),
    }
    
    angle_definitions = {
        'Knee': ("3pt", ("hip", "knee", "ankle"), True),
        'Hip':  ("3pt", ("shoulder", "hip", "knee"), False),
        'Ankle': ("4pt", ("ankle", "knee", "heel", "big_toe"), True),
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

        participants = participants[:1]
        for participant in participants:
            
            trials_path = os.path.join(path_keypoints, "csv", model_name, participant)
            trials = os.listdir(trials_path)
            trials = trials[2:3]
            for trial in trials:
                # 1) Load data
                keypoint_path = os.path.join(path_keypoints, "csv", model_name, participant, trial)
                reindexed_path = os.path.join(path_keypoints, "csv_reindexed", model_name, participant, trial)
                cleaned_path = os.path.join(path_keypoints, "csv_cleaned", model_name, participant, trial)
                y_keys = ["right_ankle_y", "right_knee_y", "right_hip_y", "shoulder_y"]

                video_path = os.path.join(path_videos, participant, "Cut", f"{trial[:-4]}.mp4")

                # temporary copy the video to avoid network issues
                tmp_video_path = os.path.join(os.path.dirname(__file__), f"tmp_{trial[:-4]}.mp4")
                shutil.copy(video_path, tmp_video_path)
                video_path = tmp_video_path


                # Initialize visualizer
                visualizer = SkeletonVisualizer(skeleton_definition=skeleton, keypoints_definition=keypoint_dict)

                
                '''
                # 1) Load raw data
                df_raw = reader.load_csv(keypoint_path)

                # 1.1) Load TD_TO
                first_event = get_first_event(trial) - 10
                
                # 1.2) Cut data
                df_raw = df_raw[df_raw["frame"] >= first_event]

                # 2) Track persons
                tracker = PersonTracker(max_distance=100, max_age=100, velocity_history=30)
                df_tracked = tracker.track(df_raw)

                #visualizer.visualize_video(video_path, df_tracked)


                # 2.1) Manual tracking rerassignment
                y_keys = ["right_ankle_y", "right_knee_y", "right_hip_y"]
                tool = ReindexingTool(df_tracked, video_path, visualizer, y_keys=y_keys, default_y_key='right_ankle_y')
                df_corrected = tool.run()

                # delete all non person_idx == 0 rows
                df_corrected = df_corrected[df_corrected["person_idx"] == 0]
                #visualizer.visualize_video(video_path, df_corrected)


                # 2.2) Save reindexed data
                reader.save_csv(df_corrected, reindexed_path)
                
                df_corrected = reader.load_csv(reindexed_path)
                #visualizer.visualize_video(video_path, df_corrected)

                # 3) Outlier detection
                detector = OutlierDetector(derivative_threshold=2.8, rolling_window_size=20, rolling_k=2.8)
                df_cleaned = detector.clean(df_corrected)

                #visualizer.visualize_video(video_path, df_cleaned)

                tool = OutlierCleaningTool(df_cleaned, video_path, visualizer, y_keys)
                df_cleaned = tool.run()

                # 3.1 Save cleaned data
                reader.save_csv(df_cleaned, cleaned_path)
                '''
                df_cleaned = reader.load_csv(cleaned_path)

                # 4 Fillter Data
                detector = OutlierDetector(derivative_threshold=2.8, rolling_window_size=20, rolling_k=2.8)
                df_filtered = detector.smooth_data(df_cleaned, video_path, cutoff=3.0, order=2)
                
                #visualizer.visualize_video(video_path, df_filtered)

                # 5) Angle calculation
                df_angles = angle_calculator.getangles(df_filtered, side="right")
                #angle_calculator.plot_angles(df_angles)
                #angle_calculator.vis_angles(df_filtered, df_angles, video_path)

                # 6) Save final
                reader.save_csv(df_angles, os.path.join("angles", model_name, participant, trial))

                
                
                #Delete tmp video
                if os.path.exists(tmp_video_path):
                    os.remove(tmp_video_path)

            #combine participant angles
            angles_df_combined, summary_df = distinct_angles(participant, model_name, reader)

            # Save combined angles
            reader.save_csv(angles_df_combined, os.path.join("angles", model_name, participant, "TD_TO_combined.csv"))
            reader.save_csv(summary_df, os.path.join("angles", model_name, participant, "summary.csv"))
            
                


if __name__ == "__main__":
    main()  