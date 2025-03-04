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
        "alphapose": AlphaPoseReader(),
    }
    
    angle_definitions = {
        'Knee': ("3pt", ("hip", "knee", "ankle"), True),
        'Hip':  ("3pt", ("shoulder", "hip", "knee"), False),
        'Ankle': ("4pt", ("ankle", "knee", "heel", "big_toe"), True),
    }

    angle_calculator = AngleCalculator(angle_definitions)

    


    for model_name, reader in model_readers.items():
        print(f"Processing: {model_name}")
        if model_name == "openpose":
            from config.keypoint_dict import openpose_keypoints as keypoint_dict
            from config.keypoint_dict import openpose_skeleton as skeleton
        elif model_name == "alphapose":
            from config.keypoint_dict import alphapose_keypoints as keypoint_dict
            from config.keypoint_dict import alphapose_skeleton as skeleton
            
        participants = [i for i in os.listdir(os.path.join(path_keypoints, "csv", model_name)) if i.startswith("P")]
        participants = sorted(participants)
        #participants = participants[:1]
        for participant in participants:
            
            trials_path = os.path.join(path_keypoints, "csv", model_name, participant)
            trials = os.listdir(trials_path)
            #trials = trials[4:5]
            video_path = "/Users/davidjaehrling/Projects/Forensic biomechanics/Participants/P01/Cut/P01_Med_Side_1.mp4"
            for trial in trials:
                # 1) Load data
                keypoint_path = os.path.join(path_keypoints, "csv", model_name, participant, trial)
                reindexed_path = os.path.join(path_keypoints, "csv_reindexed", model_name, participant, trial)
                cleaned_path = os.path.join(path_keypoints, "csv_cleaned", model_name, participant, trial)
                
                #if os.path.exists(reindexed_path):
                #   continue
                print(f"Processing: {model_name} - {participant} - {trial}")

                '''
                video_path = os.path.join(path_videos, participant, "Cut", f"{trial[:-4]}.mp4")
                # temporary copy the video to avoid network issues
                tmp_video_path = os.path.join(os.path.dirname(__file__), f"tmp_{trial[:-4]}.mp4")
                shutil.copy(video_path, tmp_video_path)
                video_path = tmp_video_path


                # Initialize visualizer
                visualizer = SkeletonVisualizer(skeleton_definition=skeleton, keypoints_definition=keypoint_dict)
                # Y keys for manual correction plotting
                y_keys = ["right_ankle_y", "right_knee_y", "right_hip_y", "right_ankle_x", "right_knee_x", "right_hip_x"]
                
                
                # 1) Load raw data
                df_raw = reader.load_csv(keypoint_path)

                # 1.1) Load TD_TO
                first_event = get_first_event(trial) - 10
                
                # 1.2) Cut data
                df = df_raw[df_raw["frame"] >= first_event]

                # 2) Track persons
                tracker = PersonTracker(max_distance=100, max_age=100, velocity_history=30)
                df = tracker.track(df)


                # 2.1) Manual tracking rerassignment
                tool = ReindexingTool(df, video_path, visualizer, y_keys=y_keys, default_y_key='right_ankle_y')
                df = tool.run()

                # delete all non person_idx == 0 rows
                df = df[df["person_idx"] == 0]


                # 2.2) Save reindexed data
                reader.save_csv(df, reindexed_path)
                df = reader.load_csv(reindexed_path)

                # 3) Outlier detection
                detector = OutlierDetector(derivative_threshold=2.9, rolling_window_size=20, rolling_k=2.8)
                df_cleaned = detector.clean(df)


                # 3.1) Manual cleaning
                tool = OutlierCleaningTool(df, video_path, visualizer, y_keys)
                df = tool.run()

                # 3.2 Save cleaned data
                reader.save_csv(df, cleaned_path)
                '''

                df = reader.load_csv(cleaned_path)

                # 4 Fillter Data
                detector = OutlierDetector(derivative_threshold=2.9, rolling_window_size=20, rolling_k=2.8)
                df = detector.int_zero(df)
                df = detector.smooth_data(df, video_path, cutoff=4.0, order=2)

                # 5) Angle calculation
                df_angles = angle_calculator.getangles(df, side="right")
                #angle_calculator.plot_angles(df_angles)
                #angle_calculator.vis_angles(df, df_angles, video_path)

                # 6) Save final
                reader.save_csv(df_angles, os.path.join("angles", model_name, participant, trial))
                
 

                

                #Delete tmp video
                #if os.path.exists(tmp_video_path):
                 #   os.remove(tmp_video_path)


            #combine participant angles
            angles_df_combined, summary_df = distinct_angles(participant, model_name, reader)

            # Save combined angles
            reader.save_csv(angles_df_combined, os.path.join("angles", model_name, participant, "TD_TO_combined.csv"))
            reader.save_csv(summary_df, os.path.join("angles", model_name, participant, "summary.csv"))
            
                


if __name__ == "__main__":
    main()  