import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from utils.tdto import get_TD_TO

class AngleCalculator:
    """
    Computes angles (e.g. knee, hip, ankle) from a DataFrame of keypoint coordinates.
    
    The angle definitions dictionary should have the following structure:
    
        angle_definitions = {
            'Knee':  ('3pt', ('hip', 'knee', 'ankle'), True),
            'Hip':   ('3pt', ('shoulder', 'hip', 'knee'), False),
            'Ankle': ('4pt', ('ankle', 'knee', 'heel', 'big_toe'), True)
        }
    
    The three-element tuple consists of:
      - the angle type ("3pt" or "4pt"),
      - a tuple of joint names,
      - a boolean flag for whether to use a clockwise sign convention.
    
    This class assumes that the DataFrame contains columns named like:
       "right_hip_x", "right_hip_y", "right_knee_x", "right_knee_y", etc.
    """
    def __init__(self, angle_definitions):
        self.angle_definitions = angle_definitions

    def getangles(self, df: pd.DataFrame, side: str = 'right') -> pd.DataFrame:
        """
        Calculate angles for each frame in the DataFrame.
        
        Parameters:
        df   : pd.DataFrame
                Input DataFrame with columns for joint coordinates.
        side : str (default 'right')
                Which side's joints to use (e.g. 'right' or 'left').
        
        Returns:
        A new DataFrame that contains non-coordinate columns (e.g. frame, person_idx, etc.)
        along with additional columns for each angle defined. The coordinate columns
        (ending in '_x' or '_y') are removed.
        """
        # Work on a copy so as not to modify the original DataFrame.
        df_out = df.copy()
        
        for angle_name, angle_info in self.angle_definitions.items():
            angle_type = angle_info[0]
            joints = angle_info[1]
            # If a clockwise flag is provided, use it; otherwise default to False.
            clockwise = angle_info[2] if len(angle_info) > 2 else False
            
            if angle_type == '3pt':
                df_out[angle_name] = df_out.apply(
                    lambda row: self._angle_3pt(row, side, joints, clockwise),
                    axis=1
                )
            elif angle_type == '4pt':
                df_out[angle_name] = df_out.apply(
                    lambda row: self._angle_4pt(row, side, joints, clockwise),
                    axis=1
                )
            else:
                # Unknown angle type: fill with NaN.
                df_out[angle_name] = np.nan
        
        # Remove coordinate columns from the output DataFrame.
        coord_cols = [col for col in df_out.columns if col.endswith('_x') or col.endswith('_y') or col.endswith('_c') or col.endswith('person_height') or col.endswith('person_idx')]
        df_out = df_out.drop(columns=coord_cols)
        
        return df_out


    @staticmethod
    def fetch_point(row, side, joint_name):
        """
        Helper to fetch (x, y) coordinates from the row.
        
        For example, if side is 'right' and joint_name is 'hip', this
        returns (row['right_hip_x'], row['right_hip_y']). If the keypoint is missing
        or zero (often indicating a failed detection), returns (np.nan, np.nan).
        """
        x_col = f"{side}_{joint_name}_x"
        y_col = f"{side}_{joint_name}_y"
        
        # If the expected columns are missing, return NaN.
        if x_col not in row or y_col not in row:
            return (np.nan, np.nan)
        
        x_val = row[x_col]
        y_val = row[y_col]
        
        # Treat 0 (or null) values as missing.
        if pd.isnull(x_val) or pd.isnull(y_val) or x_val == 0 or y_val == 0:
            return (np.nan, np.nan)
        
        return (float(x_val), float(y_val))

    @staticmethod
    def _angle_3pt(row, side, joints, clockwise=False):
        """
        Compute a 3-point angle at the middle joint.
        
        joints: a tuple of three joint names (e.g. ('hip', 'knee', 'ankle')).
        Uses the biomechanical convention:
        
          biomechanical_angle = 180 - degrees(angle)  if angle is nonnegative,
          else -(180 + degrees(angle))
        
        The `clockwise` flag (if True) may invert the sign.
        """
        jA, jB, jC = joints
        A = AngleCalculator.fetch_point(row, side, jA)
        B = AngleCalculator.fetch_point(row, side, jB)
        C = AngleCalculator.fetch_point(row, side, jC)
        
        # If any coordinate is missing, return NaN.
        if any(np.isnan(coord) for point in (A, B, C) for coord in point):
            return np.nan
        
        # Vectors BA and BC (with B as the vertex).
        BA = np.array([A[0] - B[0], A[1] - B[1]])
        BC = np.array([C[0] - B[0], C[1] - B[1]])
        
        magBA = np.linalg.norm(BA)
        magBC = np.linalg.norm(BC)
        if magBA == 0 or magBC == 0:
            return np.nan
        
        dot_val = np.dot(BA, BC)
        cos_angle = dot_val / (magBA * magBC)
        # Clamp for numerical safety.
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_radians = math.acos(cos_angle)
        
        # Determine sign using the cross product.
        cross_val = BA[0]*BC[1] - BA[1]*BC[0]
        if clockwise:
            angle_radians = -angle_radians
        if cross_val < 0:
            angle_radians = -angle_radians
        
        angle_deg = math.degrees(angle_radians)
        biomech_angle = 180 - angle_deg if angle_radians >= 0 else -(180 + angle_deg)
        return biomech_angle

    @staticmethod
    def _angle_4pt(row, side, joints, clockwise=False):
        """
        Compute a 4-point angle between two segments defined by two pairs of joints.
        
        joints: a tuple (A1, A2, B1, B2) defining two segments:
            Segment A = A2 - A1 and Segment B = B2 - B1.
        
        Here we use a biomechanical conversion:
        
            biomechanical_angle = 90 - degrees(angle)  if angle is nonnegative,
            else -(90 + degrees(angle))
        
        The `clockwise` flag (if True) may invert the sign.
        """
        A1, A2, B1, B2 = joints
        a1 = AngleCalculator.fetch_point(row, side, A1)
        a2 = AngleCalculator.fetch_point(row, side, A2)
        b1 = AngleCalculator.fetch_point(row, side, B1)
        b2 = AngleCalculator.fetch_point(row, side, B2)
        
        # If any coordinate is missing, return NaN.
        if any(np.isnan(coord) for point in (a1, a2, b1, b2) for coord in point):
            return np.nan
        
        A = np.array([a2[0] - a1[0], a2[1] - a1[1]])
        B = np.array([b2[0] - b1[0], b2[1] - b1[1]])
        
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)
        if magA == 0 or magB == 0:
            return np.nan
        
        dot_val = np.dot(A, B)
        cos_angle = dot_val / (magA * magB)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_radians = math.acos(cos_angle)
        
        cross_val = A[0]*B[1] - A[1]*B[0]
        if clockwise:
            angle_radians = -angle_radians
        if cross_val < 0:
            angle_radians = -angle_radians
        
        angle_deg = math.degrees(angle_radians)
        biomech_angle = 90 - angle_deg if angle_radians >= 0 else -(90 + angle_deg)
        return biomech_angle

    def plot_angles(self, angles):
        """
        Plot the angles over time.
        
        Parameters:
        angles : pd.DataFrame
            DataFrame with columns for each angle and a 'frame' column.
        """
        
        for col in angles.columns:
            if col == 'frame' or col == 'video_name':
                continue
            plt.plot(angles['frame'], angles[col], label=col)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.show()
    
    def vis_angles(self, df: pd.DataFrame, angles: pd.DataFrame, video_path: str):
        """
        Visualize the angles overlaid on the video.
        
        For each row in the input DataFrame (which must include a 'frame' column,
        computed angle columns, and the necessary keypoint coordinate columns), this method:
          - Opens the video from video_path.
          - For each frame indicated in the DataFrame, retrieves the corresponding video frame.
          - For each angle defined in self.angle_definitions, draws the keypoints (as circles),
            draws connecting lines between the keypoints, and overlays the computed angle value.
          - Displays the frame until a key is pressed (press ESC to exit).
        
        Parameters:
          df        : pd.DataFrame	
                        - a 'frame' column,
                        - keypoint coordinate columns (e.g. 'right_hip_x', 'right_hip_y', etc.).
          angles    : pd.DataFrame
                      DataFrame containing:
                        - a 'frame' column,
                        - computed angle columns (e.g. 'Knee', 'Hip', etc.),
                        
          video_path: str
                      Path to the video file.
        """
        side = 'right'
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return
        
        for idx, row in angles.iterrows():
            frame_num = int(row['frame'])
            row_df = df[df['frame'] == frame_num].iloc[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_num}.")
                continue

            # For each angle defined, plot its keypoints and overlay the angle value.
            for angle_name, angle_info in self.angle_definitions.items():
                angle_value = row.get(angle_name, np.nan)
                if pd.isna(angle_value):
                    continue

                angle_type = angle_info[0]
                joints = angle_info[1]
                
                # Fetch the points for the joints.
                points = []
                for joint in joints:
                    pt = self.fetch_point(row_df, side, joint)
                    if not any(np.isnan(coord) for coord in pt):
                        points.append(pt)
                
                # Draw circles for each keypoint.
                for pt in points:
                    center = (int(pt[0]), int(pt[1]))
                    cv2.circle(frame, center, radius=5, color=(0, 255, 0), thickness=-1)
                
                # Draw lines connecting the keypoints.
                if len(points) >= 2:
                    for i in range(len(points)-1):
                        pt1 = (int(points[i][0]), int(points[i][1]))
                        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                        cv2.line(frame, pt1, pt2, color=(255, 0, 0), thickness=2)
                
                # Overlay the computed angle near the middle point.
                if points:
                    mid_index = len(points) // 2
                    mid_pt = (int(points[mid_index][0]), int(points[mid_index][1]))
                    text = f"{angle_name}: {angle_value:.1f}"
                    cv2.putText(frame, text, (mid_pt[0] + 10, mid_pt[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Optionally, display the frame number.
            cv2.putText(frame, f"Frame: {frame_num}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frame = cv2.resize(frame, (width // 2, height // 2))
            cv2.imshow("Angle Visualization", frame)

            key = cv2.waitKey(1) 
            if key == 27:  # ESC key pressed
                break
        
        cap.release()
        cv2.destroyAllWindows()

        