import os
import matplotlib.pyplot as plt
import glob


def generate_superplot(results_dir="results"):
    """
    Creates a superplot combining all ROC and regression plots from the results directory.
    - ROC plots (AUC-based) are grouped together.
    - Regression plots (RMSE-based) are grouped together.
    - Titles are updated based on 'all_data' or 'before_pregnancy' folders.
    - Displays and saves the final image as 'results/superplot.png'.
    """
    # Scan for all plots in results/
    all_plots = glob.glob(os.path.join(results_dir, "**", "*.png"), recursive=True)

    # Categorize plots
    roc_plots = [p for p in all_plots if "roc_" in os.path.basename(p)]
    regression_plots = [p for p in all_plots if "pred_vs_actual" in os.path.basename(p)]

    # Sort plots alphabetically (ensures order: GDM, PE/PH, BW, BW %, etc.)
    roc_plots.sort()
    regression_plots.sort()

    # Merge lists (ROC plots first, then regression)
    all_plots_sorted = roc_plots + regression_plots

    # Determine grid layout
    num_plots = len(all_plots_sorted)
    cols = 4  # 4 plots per row
    rows = (num_plots + cols - 1) // cols  # Round up

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Plot each subplot
    for i, plot_path in enumerate(all_plots_sorted):
        img = plt.imread(plot_path)
        axes[i].imshow(img)
        axes[i].axis("off")  # Hide axes

        # Extract relevant information from file path
        parts = plot_path.split(os.sep)  # Split path into parts
        model_name = parts[-3]  # Model folder name (e.g., "GDM", "BW %", etc.)
        dataset_type = parts[-2]  # "all_data" or "before_pregnancy"

        # Determine title based on dataset type
        if dataset_type == "all_data":
            title_suffix = " (with all data)"
        else:
            title_suffix = " (with pre pregnancy data)"

        # Set title
        axes[i].set_title(f"{model_name} {title_suffix}")

    # Remove extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Save and show the superplot
    plt.tight_layout()
    superplot_path = os.path.join(results_dir, "superplot.png")
    plt.savefig(superplot_path)
    plt.show()
    print(f"Superplot saved to {superplot_path}")


# Call this at the end of main.py
if __name__ == "__main__":
    # Generate Superplot after running all models
    generate_superplot()
