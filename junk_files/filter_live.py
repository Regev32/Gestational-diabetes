import pandas as pd


def filter_live_birth_inplace(file_path):
    """
    Reads the CSV file at file_path (e.g., "data/gdm_master.csv"), filters the rows so that
    only those with Outcome == "Live birth" remain, removes the 'Outcome' column entirely,
    and saves the filtered data back to the same CSV file.

    WARNING: This function overwrites the existing CSV file. Make sure you have a backup!
    """
    df = pd.read_csv(file_path, low_memory=False)
    # Keep only rows where Outcome == "Live birth"
    df_filtered = df[df["Outcome"] == "Live birth"].copy()
    # Drop the Outcome column altogether
    df_filtered.drop(columns=["Outcome"], inplace=True)
    # Overwrite the original CSV file
    df_filtered.to_csv(file_path, index=False)
    print(f"Filtered CSV saved back to '{file_path}'.")

