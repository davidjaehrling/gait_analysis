import pandas as pd
import numpy as np
import math

class AngleCalculator:
    """
    Given a DataFrame with columns [frame, side_joint_x, side_joint_y, ...],
    compute angles (hip, knee, ankle, etc.).
    """
    def __init__(self, angle_definitions):
        """
        angle_definitions: dict describing which joints form each angle
        e.g.:
        {
          "knee_angle": ("3pt", ("hip", "knee", "ankle")), 
          "hip_angle": ("3pt", ("shoulder", "hip", "knee"))
        }
        """
        self.angle_definitions = angle_definitions

    def compute_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns like `knee_angle` to df for each frame.
        """
        for angle_name, angle_info in self.angle_definitions.items():
            angle_type = angle_info[0]
            joints = angle_info[1]

            if angle_type == '3pt':
                jA, jB, jC = joints
                df[angle_name] = df.apply(lambda row: 
                    self._3point_angle(row, jA, jB, jC), axis=1)

            elif angle_type == '4pt':
                A1, A2, B1, B2 = joints
                df[angle_name] = df.apply(lambda row: 
                    self._4point_angle(row, A1, A2, B1, B2), axis=1)
        return df

    def _3point_angle(self, row, jA, jB, jC):
        """
        For example, jA='right_hip', jB='right_knee', jC='right_ankle',
        row must have columns {right_hip_x, right_hip_y, ...}.
        """
        Ax, Ay = row.get(f"{jA}_x", np.nan), row.get(f"{jA}_y", np.nan)
        Bx, By = row.get(f"{jB}_x", np.nan), row.get(f"{jB}_y", np.nan)
        Cx, Cy = row.get(f"{jC}_x", np.nan), row.get(f"{jC}_y", np.nan)

        if any(np.isnan([Ax, Ay, Bx, By, Cx, Cy])):
            return np.nan

        # Compute angle ABC
        AB = (Ax - Bx, Ay - By)
        CB = (Cx - Bx, Cy - By)
        return self._angle_between_vectors(AB, CB)

    def _4point_angle(self, row, A1, A2, B1, B2):
        """
        If you define an angle between two segments [A1, A2] and [B1, B2].
        """
        # Implementation is up to you
        return 0.0

    def _angle_between_vectors(self, v1, v2):
        """
        Return the angle (in degrees) between vectors v1 and v2
        """
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1*mag2 == 0:
            return np.nan
        cos_ = dot/(mag1*mag2)
        # numerical safety
        cos_ = max(-1.0, min(1.0, cos_))
        angle_rad = math.acos(cos_)
        return math.degrees(angle_rad)

