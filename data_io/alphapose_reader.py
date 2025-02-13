import os
import json
import numpy as np
import pandas as pd
from data_io.base_reader import BaseReader
from config.keypoint_dict import alphapose_keypoints

class AlphaPoseReader(BaseReader):
    def load_json(self, json_dir: str) -> pd.DataFrame:
        """
        Reads multiple AlphaPose JSONs, merges them into a single DataFrame,
        with columns named consistently with alphapose_keypoints.
        """
        
        
        rows = []
        files = sorted(os.listdir(json_dir))

        for file in files:
            if not file.endswith('.json'):
                continue
            
            
            with open(os.path.join(json_dir, file), 'r') as f:
                data = json.load(f)

            videos = {}
            for entry in data:
                video_name = entry['image_id'].split('_frame')[0]
                frame_number = int(entry['image_id'].split('frame')[-1].split('.')[0])

                if video_name not in videos:
                    videos[video_name] = []

                videos[video_name].append({"frame": frame_number, **entry})

            
            # Process each video
            for video_name, entries in videos.items():
                # Sort frames by frame number
                entries.sort(key=lambda x: x['frame'])

                
                last_frame = 0
                person_idx = 0

                for entry in entries:
                    frame = entry['frame']
                
                    keypoints = entry['keypoints']

                    # Reshape keypoints to (num_keypoints, 3)
                    num_keypoints = len(keypoints) // 3
                    reshaped_keypoints = np.array(keypoints).reshape(num_keypoints, 3)


                    # Initialize person indices for the first frame
                    if frame == last_frame:
                        person_idx = person_idx + 1
                    else:
                        person_idx = 0
                        last_frame = frame

                    # Save the frame data
                    row = {
                        "video_name": video_name,
                        "frame": frame,
                        "person_idx": person_idx
                    }
                    for idx, keypoint in enumerate(reshaped_keypoints):
                        keypoint_name = alphapose_keypoints.get(idx, f'kp_{idx}')
                        row[f"{keypoint_name}_x"] = keypoint[0]
                        row[f"{keypoint_name}_y"] = keypoint[1]
                        row[f"{keypoint_name}_c"] = keypoint[2]

                    rows.append(row)
        return pd.DataFrame(rows)
    
    def load_csv(self, input_path):
        return super().load_csv(input_path)
    
    def save_csv(self, df, output_path):
        super().save_csv(df, output_path)