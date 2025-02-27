import pandas as pd

def fill_na_with_median_in_csv(csv_path: str) -> None:
    """
    Reads a CSV from 'csv_path', fills empty (NaN) cells in numeric columns with the column's median,
    and overwrites the same CSV file with these changes.

    :param csv_path: Path to the CSV file.
    """
    # 1. Read the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # 2. For each numeric column, fill NaNs with the median
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 3. Save the updated DataFrame back to the same CSV path
    df.to_csv(csv_path, index=False)


fill_na_with_median_in_csv('data/gdm_4000.csv')