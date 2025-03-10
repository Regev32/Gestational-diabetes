import pandas as pd


def fill_missing_in_file(file_path: str, exclude: list[str], method: str = "median") -> None:
    """
    Loads a CSV file from the given path, fills missing values in all numeric columns
    (except those in the 'exclude' list) using the specified method (median or average),
    and overwrites the CSV file with the updated DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.
        exclude (list[str]): List of column names to exclude from filling.
        method (str): "median" to fill with median, "avg" to fill with average. Default is "median".
    """
    # Load CSV into DataFrame
    df = pd.read_csv(file_path, low_memory=False)

    # Validate method
    if method not in ["median", "avg"]:
        raise ValueError("Method must be either 'median' or 'avg'")

    # Process each column not in the exclude list and that is numeric
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if method == "median":
                central_val = df[col].median()
            else:
                central_val = df[col].mean()
            df[col].fillna(central_val, inplace=True)

    # Overwrite the original CSV file with the updated DataFrame
    df.to_csv(file_path, index=False)


# Example usage:
if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "../data/filled_avg/gdm_4000.csv"

    # Specify columns to exclude from filling
    columns_to_exclude = ["GDM GA", "GDM drugs"]

    # Fill missing values using median (or change method to "avg" if preferred)
    fill_missing_in_file(csv_path, exclude=columns_to_exclude, method="median")

    print(f"Missing values filled and file '{csv_path}' overwritten.")
