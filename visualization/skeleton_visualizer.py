import cv2
import pandas as pd
import numpy as np

class SkeletonVisualizer:
    """
    Class that handles drawing skeletons on video frames.
    """
    def __init__(self, skeleton_definition, keypoints_definition):
        self.skeleton_definition = skeleton_definition
        self.keypoints_definition = keypoints_definition

    def draw_skeleton(self, frame, df_frame: pd.DataFrame, person_idx: int,):
        """
        Draws the skeleton of a single person on the given frame 
        according to the skeleton + keypoints definitions.
        """
        keypoint_dict = self.keypoints_definition
        skeleton = self.skeleton_definition

        person = df_frame.loc[person_idx]

        color = (0, 255, 0)
        # Define colors for right and left keypoints
        right_color = (0, 0, 255)  # Red for right
        left_color = (255, 0, 0)   # Blue for left

        # Draw keypoints
        for keypoint in keypoint_dict.values():
            if not np.isnan(person[f"{keypoint}_x"]) and not np.isnan(person[f"{keypoint}_y"]):
                x, y = int(person[f"{keypoint}_x"]), int(person[f"{keypoint}_y"])
                
                if person[f"{keypoint}_c"] > 0.1:
                    
                    if 'right' in keypoint:
                        cv2.circle(frame, (x, y), 5, right_color, -1)
                    elif 'left' in keypoint:
                        cv2.circle(frame, (x, y), 5, left_color, -1)
                    else:
                        cv2.circle(frame, (x, y), 5, color, -1)

        # Draw person_idx
        if not np.isnan(person['right_ear_x']) and not np.isnan(person['right_ear_y']):   # Check if the nose keypoint is present
            cv2.putText(frame, f"ID {person_idx}", 
                        (int(person['right_ear_x']), int(person['right_ear_y'] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Draw skeletons
            for kp1, kp2 in skeleton:
                x1, y1 = int(person[f"{kp1}_x"]), int(person[f"{kp1}_y"])
                x2, y2 = int(person[f"{kp2}_x"]), int(person[f"{kp2}_y"])
                if person[f"{kp1}_c"] > 0.1 and person[f"{kp2}_c"] > 0.1:
                    if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                        if 'right' in kp1 or 'right' in kp2:
                            cv2.line(frame, (x1, y1), (x2, y2), right_color, 2)
                        elif 'left' in kp1 or 'left' in kp2:
                            cv2.line(frame, (x1, y1), (x2, y2), left_color, 2)
                        else:
                            cv2.line(frame, (x1, y1), (x2, y2), color, 2)


    def visualize_video(self, video_path: str, df: pd.DataFrame, person_idx: int = 0):
        """
        Displays the video frame-by-frame with skeleton overlays for the chosen person_idx.
        """
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        trialname = video_path.split("\\")[-1].split(".")[0]

        # ...
        # For each frame read, filter df by that frame, draw skeleton, show image
        # ...

        for i in range(1, total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Filter df by the current frame
            df_frame = df[df["frame"] == i]

            # Draw the skeleton for all persons
            for person_idx, person in df_frame.iterrows():
                self.draw_skeleton(frame, df_frame, person_idx)

            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, trialname, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "right", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "left", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Resize the frame
            frame = cv2.resize(frame, (int(width / 3), int(height / 3)))
            # Display the frame
            cv2.imshow("Skeleton Visualization", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
