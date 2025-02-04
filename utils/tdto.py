from utils.paths import path_td_to
import pandas as pd
import os
import numpy as np





def get_TD_TO(trialname):
    data = pd.read_excel(path_td_to)

    data = data[data['Name Crop Video'] == trialname]
    # Extract all touchdown (TD) and toe-off (TO) frames into separate lists
    td_frames = []
    to_frames = []

    # Iterate over rows to collect TD and TO frames
    for index, row in data.iterrows():
        td = [row[f'Frame_TD{i}'] for i in range(1, 5) if not pd.isna(row.get(f'Frame_TD{i}', None))]
        to = [row[f'Frame_TO{i}'] for i in range(1, 5) if not pd.isna(row.get(f'Frame_TO{i}', None))]
        td_frames.append(td)
        to_frames.append(to)
        
    return td_frames[0], to_frames[0]

def get_first_event(trialname):
    trialname = trialname.split(".")[0]
    td_frames, to_frames = get_TD_TO(trialname)
    return min(td_frames[0], to_frames[0])


def distinct_angles(participant, model_name, reader):
    """
    For a given participant and model, extract distinct touchdown (TD) and toe‐off (TO) angle values
    from saved CSV files. The angles are extracted from each trial using TD and TO frame numbers (from
    get_TD_TO), and then arranged into a single row per trial with columns like:
      TD_Kneeflexion_1, TD_Kneeflexion_2, ..., TD_Hipflexion_1, etc...
    (For frontal trials, only the Kneeadduction is extracted.)

    After extraction, the function creates a summary DataFrame that computes, for each angle category,
    the overall mean, standard deviation, minimum, maximum, and range over all TD (or TO) angles.
    
    Parameters:
      participant : str
          Participant identifier (used to locate the angles CSV files).
      model_name  : str
          Model name (used as part of the folder structure, e.g., "alphapose").
      reader      : object
          An object with a method `load_csv(file_path)` that returns a pandas DataFrame.
    
    Returns:
      tuple: (angles_df_combined, summary_df)
        - angles_df_combined: DataFrame where each row corresponds to one trial and columns are distinct angle
          values (e.g. TD_Kneeflexion_1, TD_Hipflexion_1, ... TO_Kneeadduction_1, etc.).
        - summary_df: DataFrame where each row corresponds to an angle category (e.g. "TD_Kneeflexion",
          "TD_Hipflexion", etc.) and the columns show the overall mean, std, min, max, and range across all trials.
    """
    # Folder where angles CSV files are stored.
    trials_dir = os.path.join("angles", model_name, participant)
    trials = [i for i in os.listdir(trials_dir) if i.startswith("P")]
    
    participant_angles = []  # List to accumulate one dictionary per trial
    
    for trial in trials:
        # Assume trial is a CSV file; extract the base trial name (without extension).
        trial_name = trial.split(".")[0]
        
        # Get touchdown (TD) and toe‐off (TO) frame numbers (assumes get_TD_TO is defined elsewhere).
        td_frames, to_frames = get_TD_TO(trial_name)
        
        # Load the angles DataFrame from CSV.
        trial_path = os.path.join(trials_dir, trial)
        angles_df = reader.load_csv(trial_path)
        
        # Prepare a dictionary to store angles for this trial.
        trial_angles = {}
        
        # Check the trial type based on its name:
        if "Side" in trial:
            # Sagittal trial: extract Kneeflexion, Hipflexion, Ankledorsiflexion.
            for i, frame in enumerate(td_frames, start=1):
                row = angles_df[angles_df['frame'] == frame]
                if not row.empty:
                    row = row.iloc[0]
                    trial_angles[f"TD_Kneeflexion_{i}"] = row.get("Knee", np.nan)
                    trial_angles[f"TD_Hipflexion_{i}"]      = row.get("Hip", np.nan)
                    trial_angles[f"TD_Ankledorsiflexion_{i}"] = row.get("Ankle", np.nan)
            for i, frame in enumerate(to_frames, start=1):
                row = angles_df[angles_df['frame'] == frame]
                if not row.empty:
                    row = row.iloc[0]
                    trial_angles[f"TO_Kneeflexion_{i}"] = row.get("Knee", np.nan)
                    trial_angles[f"TO_Hipflexion_{i}"]      = row.get("Hip", np.nan)
                    trial_angles[f"TO_Ankledorsiflexion_{i}"] = row.get("Ankle", np.nan)
        elif "Front" in trial:
            # Frontal trial: extract only Kneeadduction.
            for i, frame in enumerate(td_frames, start=1):
                row = angles_df[angles_df['frame'] == frame]
                if not row.empty:
                    row = row.iloc[0]
                    trial_angles[f"TD_Kneeadduction_{i}"] = row.get("Knee", np.nan)
            for i, frame in enumerate(to_frames, start=1):
                row = angles_df[angles_df['frame'] == frame]
                if not row.empty:
                    row = row.iloc[0]
                    trial_angles[f"TO_Kneeadduction_{i}"] = row.get("Knee", np.nan)
        else:
            # Skip trial if not identified as Side or Front.
            continue
        
        # Add the angles for this trial to the participant list.
        participant_angles.append(trial_angles)
    
    # Combine all trial dictionaries into one DataFrame.
    angles_df_combined = pd.DataFrame(participant_angles)
    
    # --- Create Summary Statistics ---
    # We want to group columns by their angle category (e.g. "TD_Kneeflexion", "TD_Hipflexion", etc.)
    summary_groups = {}
    for col in angles_df_combined.columns:
        # The grouping key is the column name without the trailing underscore and number.
        if "_" in col:
            group = col.rsplit("_", 1)[0]
            summary_groups.setdefault(group, []).append(angles_df_combined[col])
    
    summary_data = {}
    for group, series_list in summary_groups.items():
        # Concatenate all series in the group into one Series and drop NaN values.
        combined_series = pd.concat(series_list, ignore_index=True).dropna()
        if not combined_series.empty:
            summary_data[group] = {
                "mean": combined_series.mean(),
                "std": combined_series.std(),
                "min": combined_series.min(),
                "max": combined_series.max(),
                "range": combined_series.max() - combined_series.min()
            }
    
    summary_df = pd.DataFrame(summary_data).T.reset_index().rename(columns={"index": "angle_name"})
    
    return angles_df_combined, summary_df

