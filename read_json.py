from data_io.openpose_reader import OpenPoseReader
from data_io.alphapose_reader import AlphaPoseReader
from utils.paths import path_keypoints
import os



def main():
    model_readers = {
        #"openpose": OpenPoseReader(),
        "alphapose": AlphaPoseReader(),
    }

    for model_name, reader in model_readers.items():
        print(f"Processing: {model_name}")
        df_raw = reader.load_json(f"{path_keypoints}\json\{model_name}")

        top_save_path = f"{path_keypoints}\csv\{model_name}"

        participants = df_raw["video_name"].str.split("_").str[0].unique()

        for participant in participants:
            print(f"Processing participant: {participant}")
            df_participant = df_raw[df_raw["video_name"].str.startswith(participant)]
            
            save_path = f"{top_save_path}\{participant}"
            # Save the participant data
            trials = df_participant["video_name"].unique()

            for trial in trials:
                print(f"Processing trial: {trial}")
                df_trial = df_participant[df_participant["video_name"].str.contains(trial)]
                # Save the trial data
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                df_trial.to_csv(f"{save_path}\{trial}.csv", index=False)

            




if __name__ == "__main__":
    main()