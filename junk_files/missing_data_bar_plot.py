import matplotlib.pyplot as plt
import pandas as pd


def plot_missing_data(df: pd.DataFrame) -> None:
    total_rows = len(df)
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print("No missing values in the DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    ax = missing_counts.plot(kind='bar', rot=0)
    plt.xlabel('Columns')
    plt.ylabel('Missing Count')
    plt.title('Missing Data by Column (Only Columns with Missing Values)')

    for i, (col, count) in enumerate(missing_counts.items()):
        label = f"{count}"
        ax.text(i, count + total_rows * 0.01, label, ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../data/filled_avg/gdm_master.csv", low_memory=False)
    # Plot missing data
    plot_missing_data(df)
