import os
import math
import pandas as pd
import matplotlib.pyplot as plt


def plot_histograms_side_by_side(file_path):
    """
    Reads the CSV file at file_path and for each column in the list
    [GDM, GDM GA, GDM drugs, Outcome, out ga, BW, BW %, PE/PH] plots a histogram.

    - Each individual histogram is saved in the "histogram" folder.
    - Then, a combined figure (superplot) is created that arranges the histograms in a grid
      with 4 plots per row, saved as "histograms_combined.png" in the "histogram" folder.
    """
    # Ensure the histogram folder exists
    hist_folder = "histogram"
    os.makedirs(hist_folder, exist_ok=True)

    # Read the CSV file (using low_memory=False to avoid dtype warnings)
    df = pd.read_csv(file_path, low_memory=False)

    # Define the columns of interest
    cols_to_plot = ["GDM", "GDM GA", "GDM drugs", "Outcome", "out ga", "BW", "BW %", "PE/PH"]

    # Filter columns that exist in the dataframe
    available_cols = [col for col in cols_to_plot if col in df.columns]

    if not available_cols:
        print("None of the specified columns were found in the CSV file.")
        return

    # Save individual histograms
    for col in available_cols:
        plt.figure(figsize=(8, 6))
        data = df[col].dropna()
        plt.hist(data, bins=50, edgecolor="black")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        # Sanitize filename and save in histogram folder
        filename = os.path.join(hist_folder, f"hist_{col.replace(' ', '_').replace('/', '_')}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved histogram for '{col}' as {filename}")

    # Create a combined superplot with 4 plots per row
    n_cols_total = len(available_cols)
    n_rows = math.ceil(n_cols_total / 4)

    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # flatten in case of multiple rows

    for i, col in enumerate(available_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=50, edgecolor="black")
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Remove any unused subplots
    for j in range(i + 1, n_rows * 4):
        fig.delaxes(axes[j])

    plt.tight_layout()
    combined_filename = os.path.join(hist_folder, "histograms_combined.png")
    plt.savefig(combined_filename)
    plt.show()
    print(f"Saved combined histogram plot as {combined_filename}")

# Example usage:
plot_histograms_side_by_side("data/gdm_master.csv")
