import pandas as pd


def filter_live(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df_filtered = df[df["Outcome"] == "Live birth"].copy()
    df_filtered.drop(columns=["Outcome"], inplace=True)
    df_filtered.to_csv(file_path, index=False)
    print(f"Filtered CSV saved back to '{file_path}'.")

filter_live("../data/filled_avg/gdm_master.csv")