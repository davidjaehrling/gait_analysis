import os
import json
import numpy as np
import pandas as pd
from data_io.base_reader import BaseReader
from config.keypoint_dict import openpose_keypoints

class OpenPoseReader(BaseReader):
    """
    Concrete reader for OpenPose JSON format.
    """
    def load_json(self, json_dir: str) -> pd.DataFrame:
        """
        Reads multiple OpenPose JSONs from `json_dir`, merges them into a single DataFrame.
        Assumes each JSON file name includes frame info.
        """
        rows = []
        files = sorted(os.listdir(json_dir))

        for file in files:
            if not file.endswith('.json'):
                continue
            
            # parse out video name and frame number from filename
            video_name = "_".join(file.split('_')[0:4])  
            frame_num = int(file.split('_')[-2][-4:])
            
            with open(os.path.join(json_dir, file), 'r') as f:
                data = json.load(f)

            # For each person in the JSON, save keypoints
            for person_idx, person in enumerate(data.get("people", [])):
                keypoints = person["pose_keypoints_2d"]
                num_kpts = len(keypoints) // 3
                reshaped = np.array(keypoints).reshape(num_kpts, 3)

                row = {
                    "video_name": video_name,
                    "frame": frame_num,
                    "person_idx": person_idx
                }
                for kpt_id, (x, y, c) in enumerate(reshaped):
                    kpt_name = openpose_keypoints.get(kpt_id, f"kp_{kpt_id}")
                    row[f"{kpt_name}_x"] = x
                    row[f"{kpt_name}_y"] = y
                    row[f"{kpt_name}_c"] = c
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def load_csv(self, input_path):
        return super().load_csv(input_path)