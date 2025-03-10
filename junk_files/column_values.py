import matplotlib.pyplot as plt
import pandas as pd


def plot_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    data = df[col].dropna()

    if pd.api.types.is_numeric_dtype(data):
        plt.figure()
        plt.hist(data, bins=30, edgecolor='black')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {col}')
        plt.tight_layout()
        plt.show()
    else:
        counts = data.value_counts()
        plt.figure()
        ax = counts.plot(kind='bar', rot=0)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Bar Chart of {col}')

        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('../data/filled_avg/gdm_master.csv')

    plot_column(df, 'out ga')

