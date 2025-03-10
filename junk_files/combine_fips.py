#!/usr/bin/env python
"""
This script loads the already-saved feature importance plots for a specified dataset
and displays them side by side. The plot from the "all_data" mode is shown on the left,
and the plot from the "not_all_data" mode is shown on the right.

It assumes that the plots are saved as:
  ../results/<dataset_name>/<model_class>/all_data/feature_importence_plot.png
  ../results/<dataset_name>/<model_class>/not_all_data/feature_importence_plot.png

Usage:
    python display_combined_plots.py <dataset_name> [--model_class MODEL_CLASS]

Example:
    python display_combined_plots.py GDM --model_class XGBClassifier
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


def display_combined_feature_importance(dataset_name, model_class="XGBClassifier"):
    base_path = os.path.join("../", "results", dataset_name, model_class)
    all_data_path = os.path.join(base_path, "all_data", "feature_importance_plot.png")
    not_all_data_path = os.path.join(base_path, "not_all_data", "feature_importance_plot.png")

    # Check if the image files exist.
    if not os.path.exists(all_data_path):
        print(f"File not found: {all_data_path}")
        return
    if not os.path.exists(not_all_data_path):
        print(f"File not found: {not_all_data_path}")
        return

    # Load images using matplotlib.
    img_all = mpimg.imread(all_data_path)
    img_not_all = mpimg.imread(not_all_data_path)

    # Create a figure with two side-by-side subplots.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_all)
    axes[0].set_title(f"{dataset_name} (with all data)")
    axes[0].axis('off')  # Hide axes for a cleaner look.

    axes[1].imshow(img_not_all)
    axes[1].set_title(f"{dataset_name} (without all data)")
    axes[1].axis('off')

    plt.suptitle(f"Feature Importance Comparison for {dataset_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config_path = os.path.join("../data", "config.json")
    with open(config_path, 'r') as f:
        datasets_config = json.load(f)

    for dataset_name, config in datasets_config.items():
        print(f"\nProcessing dataset: {dataset_name}")
        data_path = config.get("data_path")
        model_class = config.get("model_class", "XGBClassifier")
        display_combined_feature_importance(dataset_name, model_class)
