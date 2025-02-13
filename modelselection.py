import os
import cv2
import pandas as pd

# Import your readers and visualizer.
from data_io.openpose_reader import OpenPoseReader 
from data_io.alphapose_reader import AlphaPoseReader
from visualization.skeleton_visualizer import SkeletonVisualizer
from utils.paths import path_videos, path_keypoints

def main():
    participant = "P01"
    model_names = ["openpose", "alphapose"]
    
    # Dictionary to store results per model.
    results = {}
    
    # Loop over each model.
    for model in model_names:
        total_frames = 0
        bad_frames = 0
        
        if model == "openpose":
            from config.keypoint_dict import openpose_keypoints as keypoint_dict
            from config.keypoint_dict import openpose_skeleton as skeleton
        elif model == "alphapose":
            from config.keypoint_dict import alphapose_keypoints as keypoint_dict
            from config.keypoint_dict import alphapose_skeleton as skeleton

        # Build the directory path for reindexed CSV files.
        csv_dir = os.path.join(path_keypoints, "csv_reindexed", model, participant)
        if not os.path.isdir(csv_dir):
            print(f"Directory not found: {csv_dir}")
            continue
        
        # Get a list of trial CSV files.
        trial_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        
        # Create a reader instance based on the model.
        if model.lower() == "openpose":
            reader = OpenPoseReader()
        elif model.lower() == "alphapose":
            reader = AlphaPoseReader()
        else:
            print(f"Unknown model {model}")
            continue
        
        visualizer = SkeletonVisualizer(skeleton_definition=skeleton, keypoints_definition=keypoint_dict)
        
        # Process each trial.
        for trial in trial_files:
            trial_name = os.path.splitext(trial)[0]
            trial_csv_path = os.path.join(csv_dir, trial)
            
            # Load the reindexed CSV.
            df = reader.load_csv(trial_csv_path)
            if df.empty:
                continue
            
            video_path = os.path.join(path_videos, participant,"Cut", trial_name + ".mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                continue
            
            
            # Iterate over each row in the CSV (each row represents a frame with keypoints).
            for idx, row in df.iterrows():
                
                frame_number = int(row["frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    continue
                row2 = df.loc[df["frame"] == frame_number]
                # Overlay the skeleton on the frame.
                # (Assumes that visualizer.draw_skeleton takes the frame and a row or keypoints data.)
                visualizer.draw_skeleton(frame, row, 0)
                
                # Display the frame.
                cv2.putText(frame, f"Frame: {frame_number}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Model Selection", frame)
                
                # Wait for key press: g (good), b (bad), or q (quit current trial).
                key = cv2.waitKey(0) & 0xFF
                if key == ord('g'):
                    # Good frame.
                    pass
                elif key == ord('b'):
                    # Bad frame.
                    bad_frames += 1
                elif key == ord('q'):
                    # Quit this trial.
                    break
                total_frames += 1
            
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate percentage of frames needing correction.
        percent_bad = (bad_frames / total_frames * 100) if total_frames > 0 else 0
        results[model] = {"total_frames": total_frames, "bad_frames": bad_frames, "percent_bad": percent_bad}
    
    # Print out the results.
    for model, res in results.items():
        print(f"Model: {model}")
        print(f"  Total frames evaluated: {res['total_frames']}")
        print(f"  Frames needing correction: {res['bad_frames']}")
        print(f"  Percentage needing correction: {res['percent_bad']:.1f}%\n")

if __name__ == "__main__":
    main()
