import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np
import math

class PersonTracker:
    """
    Class that reassigns 'person_idx' across frames to ensure consistent identity
    using:
      - Hungarian assignment for global minimal distance,
      - A constant-velocity prediction model with velocity smoothing,
      - Optionally corridor prioritization (line1, line2),
      - Height-based matching to reduce ID swaps.
    """

    def __init__(self, 
                 max_distance=50, 
                 max_age=30, 
                 velocity_history=5,
                 corridor_line1=None,
                 corridor_line2=None,
                 alpha=1.0,
                 beta=0.1):
        """
        Parameters
        ----------
        max_distance : float
            Maximum distance allowed to match a track to a detection.
        max_age : int
            Maximum number of frames a track can remain unmatched before removal.
        velocity_history : int
            Number of most recent velocities to average for constant-velocity prediction.
        corridor_line1, corridor_line2 : tuple or None
            If provided, each is a line (A, B, C) from the equation Ax + By + C = 0.
            Used to prioritize the track that remains longest inside the corridor.
        alpha : float
            Weight for positional distance in cost.
        beta : float
            Weight for height difference in cost.
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.velocity_history = velocity_history
        self.line1 = corridor_line1
        self.line2 = corridor_line2

        # Weights for combining distance + height difference
        self.alpha = alpha
        self.beta = beta

    def track(self, df: pd.DataFrame,
              keypoint_x='right_hip_x',
              keypoint_y='right_hip_y') -> pd.DataFrame:
        """
        Reassigns person indices across frames to yield a consistent 'person_idx'.

        Parameters
        ----------
        df : pd.DataFrame
            Must include columns: ['frame', keypoint_x, keypoint_y] plus
            'right_shoulder_x','right_shoulder_y','right_ankle_x','right_ankle_y' 
            for height computation (if you want height-based matching).
        keypoint_x, keypoint_y : str
            Columns used to represent the position of a person (for matching).
        
        Returns
        -------
        pd.DataFrame
            The same data but with updated 'person_idx' reflecting consistent identity.
        """
        required_cols = {'frame', keypoint_x, keypoint_y,
                         'right_shoulder_x','right_shoulder_y','right_ankle_x','right_ankle_y'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns {required_cols} for height-based tracking.")

        # Convert DataFrame rows to a list of dicts for easier manipulation
        data = df.to_dict(orient='records')

        # 1) Group detections by frame
        frames_dict = defaultdict(list)
        for row in data:
            frame_id = int(float(row['frame']))
            frames_dict[frame_id].append(row)

        sorted_frames = sorted(frames_dict.keys())

        # We keep track of active tracks by an integer track_id -> track_info
        # track_info = {
        #   'x': current x,
        #   'y': current y,
        #   'vx': average velocity x,
        #   'vy': average velocity y,
        #   'height': last smoothed height,
        #   'history': [(frame_num, x, y), ...],
        #   'age': number_of_unmatched_frames,
        #   'recent_velocities': [(vx1, vy1), ... up to velocity_history]
        # }
        active_tracks = {}
        next_track_id = 0

        # Go through frames in order
        for frame_id in sorted_frames:
            detections = frames_dict[frame_id]

            # Build lists for each detection's main coordinate + height
            coords = []
            heights = []

            for det in detections:
                # Parse main x,y for matching
                x = self._parse_float(det.get(keypoint_x))
                y = self._parse_float(det.get(keypoint_y))

                coords.append((x, y))

                # Compute person height from right_shoulder -> right_ankle
                shx = self._parse_float(det.get('right_shoulder_x'))
                shy = self._parse_float(det.get('right_shoulder_y'))
                ax = self._parse_float(det.get('right_ankle_x'))
                ay = self._parse_float(det.get('right_ankle_y'))

                # If we can't compute, set to None
                if (shx is None or shy is None or ax is None or ay is None):
                    person_height = None
                else:
                    person_height = self._euclidean_distance(shx, shy, ax, ay)

                # Store in the detection dict, so we can retrieve later
                det['person_height'] = person_height
                heights.append(person_height)

            # 1) Predict each track's next position using averaged velocity
            track_ids = list(active_tracks.keys())
            predicted_positions = []
            for tid in track_ids:
                tinfo = active_tracks[tid]

                # Compute average velocity from recent velocities
                if len(tinfo['history']) >= 2:
                    recent_velocities = tinfo.get('recent_velocities', [])
                    if len(recent_velocities) > 0:
                        avg_vx = sum(v[0] for v in recent_velocities) / len(recent_velocities)
                        avg_vy = sum(v[1] for v in recent_velocities) / len(recent_velocities)
                    else:
                        avg_vx, avg_vy = tinfo['vx'], tinfo['vy']
                else:
                    avg_vx, avg_vy = tinfo['vx'], tinfo['vy']

                px = tinfo['x'] + avg_vx
                py = tinfo['y'] + avg_vy
                predicted_positions.append((px, py))

            # 2) Build cost matrix for the Hungarian algorithm
            cost_matrix = []
            for t_idx, (px, py) in enumerate(predicted_positions):
                tinfo = active_tracks[track_ids[t_idx]]
                track_height = tinfo.get('height', None)

                row_cost = []
                for d_idx, (dx, dy) in enumerate(coords):
                    if dx is None or dy is None:
                        # Invalid detection coordinates
                        dist = float('inf')
                    else:
                        dist = self._euclidean_distance(px, py, dx, dy)

                    # If dist > self.max_distance, we could skip,
                    # but let's incorporate height anyway:
                    cost = dist

                    # Now add a term for height difference
                    det_height = heights[d_idx]
                    if track_height is not None and det_height is not None:
                        height_diff = abs(track_height - det_height)
                        cost += self.beta * height_diff

                    row_cost.append(cost)

                cost_matrix.append(row_cost)

            # 3) Solve the assignment problem
            if cost_matrix and len(cost_matrix[0]) > 0:
                cost_matrix_np = np.array(cost_matrix)
                track_indices, detection_indices = linear_sum_assignment(cost_matrix_np)
            else:
                track_indices, detection_indices = [], []

            # 4) Filter out matches that exceed max_distance in positional sense
            matches = []
            for t_idx, d_idx in zip(track_indices, detection_indices):
                # The position-based cost is the first part of cost_matrix[t_idx][d_idx]
                # but let's do an explicit check if the actual position distance < max_distance
                px, py = predicted_positions[t_idx]
                dx, dy = coords[d_idx]
                if dx is None or dy is None:
                    continue
                pos_dist = self._euclidean_distance(px, py, dx, dy)

                if pos_dist < self.max_distance:
                    matches.append((t_idx, d_idx))

            matched_tracks = set()
            matched_detections = set()

            # Update matched tracks
            for t_idx, d_idx in matches:
                tid = track_ids[t_idx]
                dx, dy = coords[d_idx]

                tinfo = active_tracks[tid]
                old_x, old_y = tinfo['x'], tinfo['y']

                # New velocity
                new_vx = dx - old_x
                new_vy = dy - old_y

                # Keep track of velocity history
                if 'recent_velocities' not in tinfo:
                    tinfo['recent_velocities'] = []
                tinfo['recent_velocities'].append((new_vx, new_vy))
                if len(tinfo['recent_velocities']) > self.velocity_history:
                    tinfo['recent_velocities'].pop(0)

                # Recompute average velocity
                avg_vx = sum(v[0] for v in tinfo['recent_velocities']) / len(tinfo['recent_velocities'])
                avg_vy = sum(v[1] for v in tinfo['recent_velocities']) / len(tinfo['recent_velocities'])

                # Update track state
                tinfo['x'] = dx
                tinfo['y'] = dy
                tinfo['vx'] = avg_vx
                tinfo['vy'] = avg_vy
                tinfo['age'] = 0  # reset unmatched age
                tinfo['history'].append((frame_id, dx, dy))

                # Update track's height by smoothing
                det_height = detections[d_idx]['person_height']
                old_height = tinfo.get('height', None)
                if det_height is not None:
                    if old_height is None:
                        # First time we have a measurement
                        tinfo['height'] = det_height
                    else:
                        # Simple smoothing factor
                        smoothing_factor = 0.7
                        tinfo['height'] = (smoothing_factor * old_height 
                                           + (1 - smoothing_factor) * det_height)

                # Assign corrected_pid to the detection
                detections[d_idx]['corrected_pid'] = tid

                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)

            # 5) Increment age for unmatched tracks
            unmatched_track_indices = set(range(len(track_ids))) - matched_tracks
            for t_idx in unmatched_track_indices:
                tid = track_ids[t_idx]
                active_tracks[tid]['age'] += 1

            # 6) Remove inactive tracks
            to_remove = [tid for tid in active_tracks if active_tracks[tid]['age'] > self.max_age]
            for tid in to_remove:
                del active_tracks[tid]

            # 7) Create new tracks for unmatched detections
            unmatched_detection_indices = set(range(len(coords))) - matched_detections
            for d_idx in unmatched_detection_indices:
                dx, dy = coords[d_idx]
                if dx is None or dy is None:
                    # Skip invalid detections
                    continue

                # Create new track
                det_height = detections[d_idx]['person_height']

                active_tracks[next_track_id] = {
                    'x': dx,
                    'y': dy,
                    'vx': 0.0,
                    'vy': 0.0,
                    'height': det_height,  # start with detection height
                    'history': [(frame_id, dx, dy)],
                    'age': 0,
                    'recent_velocities': []
                }
                detections[d_idx]['corrected_pid'] = next_track_id
                next_track_id += 1

        # ----------------------------------------------------------------------
        # AFTER assigning corrected_pid, compute total distance for each track
        # ----------------------------------------------------------------------
        track_movement = {}
        for tid, tinfo in active_tracks.items():
            history = tinfo['history']
            history.sort(key=lambda x: x[0])  # sort by frame index
            total_dist = 0.0
            for i in range(len(history) - 1):
                _, x1, y1 = history[i]
                _, x2, y2 = history[i+1]
                if x1 is None or y1 is None or x2 is None or y2 is None:
                    continue
                total_dist += self._euclidean_distance(x1, y1, x2, y2)
            track_movement[tid] = total_dist

        #  If corridor lines are defined, prioritize the track that spends
        #  the most frames inside the corridor by artificially boosting total_dist
        if self.line1 is not None and self.line2 is not None:
            best_tid = None
            best_inliers = -1
            for tid, tinfo in active_tracks.items():
                inliers = sum(
                    1
                    for (_, x, y) in tinfo['history']
                    if x is not None and y is not None
                    and self._point_between_two_lines(x, y, self.line1, self.line2)
                )
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_tid = tid
            # "Boost" so that track gets sorted to the top
            if best_tid is not None:
                track_movement[best_tid] += 1e6

        # Sort tracks by total movement descending
        tracks_sorted = sorted(track_movement.keys(), key=lambda tid: track_movement[tid], reverse=True)

        # Map old track_id -> new stable person_idx
        track_id_map = {tid: idx for idx, tid in enumerate(tracks_sorted)}

        # 8) Build final DataFrame
        #    Each row in 'data' has a 'corrected_pid' assigned if it was matched/created.
        #    We'll set 'person_idx' based on that sorted order.
        for row in data:
            old_tid = row.get('corrected_pid', None)
            if old_tid is not None:
                row['person_idx'] = track_id_map.get(old_tid, -1)
            # else remains whatever it was (or -1 if you prefer)

        # Optionally drop the 'corrected_pid' column
        for row in data:
            if 'corrected_pid' in row:
                del row['corrected_pid']

        # Convert back to DataFrame
        final_df = pd.DataFrame(data)
        # Sort by [frame, person_idx] for convenience
        final_df.sort_values(by=['frame', 'person_idx'], inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        return final_df

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    @staticmethod
    def _euclidean_distance(x1, y1, x2, y2):
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return float('inf')
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    @staticmethod
    def _parse_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _point_between_two_lines(x, y, line1, line2):
        """
        line is given by (A, B, C) from A*x + B*y + C = 0
        return True if (x, y) is between line1 and line2 
        according to your corridor's orientation logic.
        """
        A1, B1, C1 = line1
        A2, B2, C2 = line2
        side1 = A1*x + B1*y + C1
        side2 = A2*x + B2*y + C2
        # Adjust sign logic to match your corridor orientation
        return (side1 >= 0 and side2 <= 0)
