#!/usr/bin/env python
"""
This script loads the configuration from ../data/config.json, and for each dataset (and for both modes:
"all_data" and "not_all_data") it:
  - Loads the original CSV.
  - Applies the appropriate mapping/preprocessing (e.g. for GDM, GDM_drugs, GDM_GA, out_ga, BW, BW_%, or PE_PH).
  - Splits the data and applies one-hot encoding.
  - Loads the saved model from the corresponding results folder (e.g. ../results/GDM/XGBClassifier/all_data/model.pkl).
  - Evaluates the model on the test set:
      * For XGBClassifier: If binary, it plots a ROC curve; if multi-class, it computes an overall AUC and then picks the majority class for plotting a one-vs-rest ROC curve.
      * For XGBRegressor: It computes the Pearson correlation and plots a scatter plot (actual vs. predicted).
  - Saves the evaluation plot in that results folder.
After processing all datasets/modes, it combines all evaluation plots into a single superplot.
"""

import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------
# Settings & Constants
# ---------------------------
xgb_hyperparameters = {
    "n_estimators": {"type": "int", "min": 100, "max": 3000, "distribution": "loguniform"},
    "max_depth": {"type": "int", "min": 1, "max": 11},
    "min_child_weight": {"type": "float", "min": 1, "max": 100, "distribution": "loguniform"},
    "subsample": {"type": "float", "min": 0.5, "max": 1.0},
    "learning_rate": {"type": "float", "min": 1e-5, "max": 0.7, "distribution": "loguniform"},
    "colsample_bylevel": {"type": "float", "min": 0.5, "max": 1.0},
    "colsample_bytree": {"type": "float", "min": 0.5, "max": 1.0},
    "gamma": {"type": "float", "min": 1e-8, "max": 7, "distribution": "loguniform"},
    "lambda": {"type": "float", "min": 1, "max": 4, "distribution": "loguniform"},
    "alpha": {"type": "float", "min": 1e-8, "max": 100, "distribution": "loguniform"}
}

# Column groups for feature selection:
BEFORE_COLS = [
    "Age", "Race", "Height", "Weight", "BMI 12w", "Conception", "Smoking",
    "Chronic hypertension", "Previous GDM", "Previous FGR", "Previous LGA", "Last BW%"
]
CRL_COLS = ["CRL", "Machine", "hCG", "PAPP-A", "Ut PI"]
ID_COLS = ["id sort", "Scan date"]


# ---------------------------
# Data Handling Functions
# ---------------------------
def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)


def map_gdm_to_binary(df):
    mapping = {"no": 0, "a gdm": 1}
    df["GDM"] = df["GDM"].astype(str).str.lower().str.strip().map(mapping)
    df = df[df["GDM"].notna()].copy()
    return df.rename(columns={"GDM": "y"})


def map_gdm_drugs_to_numeric(df):
    mapping = {"diet": 0, "insulin": 1, "insulin, metformin": 2, "metformin": 3}
    df["GDM_drugs"] = df["GDM_drugs"].astype(str).str.lower().str.strip()
    df["GDM_drugs"] = df["GDM_drugs"].map(mapping)
    df = df[df["GDM_drugs"].notna()].copy()
    return df.rename(columns={"GDM_drugs": "y"})


def map_others_to_categorical(df, target_col):
    df = df.dropna(subset=[target_col]).copy()
    return df.rename(columns={target_col: "y"})


def map_peph_to_numeric(df):
    mapping = {"no": 0, "a pe": 1, "b pie": 2}
    df["PE_PH"] = df["PE_PH"].astype(str).str.lower().str.strip().map(mapping)
    df = df[df["PE_PH"].notna()].copy()
    return df.rename(columns={"PE_PH": "y"})


def split_by_id(df):
    unique_ids = df["id sort"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    n_ids = len(unique_ids)
    n_train = int(0.64 * n_ids)
    n_val = int(0.16 * n_ids)
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]
    df_train = df[df["id sort"].isin(train_ids)].copy()
    df_val = df[df["id sort"].isin(val_ids)].copy()
    df_test = df[df["id sort"].isin(test_ids)].copy()
    return df_train, df_val, df_test


def one_hot_encode(df):
    exclude_cols = {"id sort", "Scan date", "y"}
    cat_cols = [col for col in df.select_dtypes(include=["object"]).columns if col not in exclude_cols]
    if cat_cols:
        return pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.copy()


# ---------------------------
# Plotting Functions
# ---------------------------
def plot_and_save_roc_curve(y_true, y_pred, save_filepath):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    delete_if_exists(save_filepath)
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regression_results(y_true, y_pred, save_filepath):
    """
    Plot a scatter plot of actual vs. predicted values for regression,
    with a reference line (y = x) to indicate perfect predictions.
    Also annotates the plot with the Pearson correlation.
    """
    corr = stats.pearsonr(y_true, y_pred)[0]
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.text(0.05, 0.95, f"Correlation = {corr:.2f}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.5))
    plt.legend()
    delete_if_exists(save_filepath)
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_auc_trials(auc_values, save_filepath):
    plt.figure()
    plt.plot(range(1, len(auc_values) + 1), auc_values, marker='o')
    plt.xlabel("Trial")
    plt.ylabel("Score")
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


def delete_if_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


# ---------------------------
# Recreate Evaluation Plots from Saved Models
# ---------------------------
def recreate_evaluation_plots(config_path="../data/config.json"):
    """
    For each dataset in the configuration file (and for both modes: "all_data" and "not_all_data"),
    load the saved model from the corresponding results folder, process the test set, evaluate, and
    re-create the evaluation plot (ROC for classifier, regression scatter for regressor).
    """
    with open(config_path, 'r') as f:
        datasets_config = json.load(f)
    for dataset_name, config in datasets_config.items():
        data_path = config.get("data_path")
        model_class = config.get("model_class", "XGBClassifier")
        # Loop over modes: all_data True/False
        for all_data in [True, False]:
            data_flag = "all_data" if all_data else "not_all_data"
            result_folder = os.path.join("..", "results", dataset_name, model_class, data_flag)
            print(
                f"\nRecreating evaluation plot for {dataset_name} ({model_class}, {data_flag}) in folder {result_folder}")

            # Load and preprocess data (using same pipeline as before)
            df = load_data(data_path)
            if dataset_name == "GDM":
                df = map_gdm_to_binary(df)
            elif dataset_name == "GDM_drugs":
                df = map_gdm_drugs_to_numeric(df)
            elif dataset_name in ["GDM_GA", "out_ga", "BW", "BW_%"]:
                df = map_others_to_categorical(df, dataset_name)
            elif dataset_name == "PE_PH":
                df = map_peph_to_numeric(df)
            else:
                print(f"Invalid dataset name: {dataset_name}")
                continue

            # Build feature set; include CRL_COLS if all_data is True.
            columns_needed = ID_COLS + BEFORE_COLS + ["y"]
            if all_data:
                columns_needed += CRL_COLS
            df = df[columns_needed].dropna(subset=["y"]).copy()

            # Split and one-hot encode
            df_train, df_val, df_test = split_by_id(df)
            for subset in [df_train, df_val, df_test]:
                subset.drop(columns=["id sort", "Scan date"], inplace=True, errors="ignore")
            df_train = one_hot_encode(df_train)
            df_val = one_hot_encode(df_val)
            df_test = one_hot_encode(df_test)
            X_test = df_test.drop(columns=["y"])
            y_test = df_test["y"]

            # Load saved model
            model_file = os.path.join(result_folder, "model.pkl")
            if not os.path.exists(model_file):
                print(f"Model file not found: {model_file}")
                continue
            with open(model_file, "rb") as f_model:
                model = pickle.load(f_model)

            # Evaluate and re-create the appropriate plot
            if model_class == "XGBClassifier":
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) > 2:
                    overall_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                    print(f"[{dataset_name} - {data_flag}] Multi-class AUC (OVR) = {overall_auc:.4f}")
                    # For plotting, choose majority class as positive
                    majority_class = y_test.value_counts().idxmax()
                    y_test_binary = (y_test == majority_class).astype(int)
                    y_proba_binary = y_proba[:, int(majority_class)]
                    plot_path = os.path.join(result_folder, "roc_curve.png")
                    plot_and_save_roc_curve(y_test_binary, y_proba_binary, plot_path)
                    print(f"Saved one-vs-rest ROC curve for class '{majority_class}' to {plot_path}")
                else:
                    y_pred = y_proba[:, 1]
                    auc_value = roc_auc_score(y_test, y_pred)
                    print(f"[{dataset_name} - {data_flag}] Binary AUC = {auc_value:.4f}")
                    plot_path = os.path.join(result_folder, "roc_curve.png")
                    plot_and_save_roc_curve(y_test, y_pred, plot_path)
                    print(f"Saved ROC curve to {plot_path}")
            elif model_class == "XGBRegressor":
                y_pred = model.predict(X_test)
                corr = stats.pearsonr(y_test, y_pred)[0]
                print(f"[{dataset_name} - {data_flag}] Correlation = {corr:.4f}")
                plot_path = os.path.join(result_folder, "regression_scatter.png")
                plot_regression_results(y_test, y_pred, plot_path)
                print(f"Saved regression scatter plot to {plot_path}")
            else:
                print(f"Unsupported model: {model_class}")


# ---------------------------
# Generate Superplot
# ---------------------------
def generate_superplot(results_dir="../results"):
    """
    Combines evaluation plots into one superplot.
    Only includes files named 'roc_curve.png' or 'regression_scatter.png'.
    Each subplot is titled in the format:
      "{plot type} for {dataset} using {model_class} {with all data/without all data}"
    The final image is saved as 'superplot.png' in the results directory.
    """
    all_plots = glob.glob(os.path.join(results_dir, "**", "*.png"), recursive=True)
    relevant_plots = []
    for p in all_plots:
        filename = os.path.basename(p)
        if filename in ["roc_curve.png", "regression_scatter.png"]:
            relevant_plots.append(p)
    if not relevant_plots:
        print(f"No evaluation plots found in '{results_dir}'. Exiting superplot generation.")
        return
    relevant_plots.sort()
    num_plots = len(relevant_plots)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, plot_path in enumerate(relevant_plots):
        img = plt.imread(plot_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        parts = plot_path.split(os.sep)
        # Expected structure: results/{dataset}/{model_class}/{data_flag}/{filename}
        dataset = parts[-4]
        model_cls = parts[-3]
        data_flag = parts[-2]
        filename = parts[-1]
        plot_type = "ROC curve" if filename == "roc_curve.png" else "Regression scatter"
        data_text = "with all data" if data_flag == "all_data" else "without all data"
        title = f"{plot_type} for {dataset} using {model_cls} {data_text}"
        axes[i].set_title(title, fontsize=9)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    superplot_path = os.path.join(results_dir, "superplot.png")
    plt.savefig(superplot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Superplot saved to {superplot_path}")


# ---------------------------
# Generate Superplot2 (Horizontal Bar Plots)
# ---------------------------
def generate_superplot2(results_dir="../results", config_path="../data/config.json"):
    """
    Iterates over all datasets and modes from the configuration, loads each saved model and its test data,
    computes a performance metric (AUC for XGBClassifier and Pearson correlation for XGBRegressor), and then
    creates a horizontal bar plot with:
      - Y-axis labels that are just the dataset name (with mode indicated in parentheses).
      - A title that includes "XGB" and the evaluation context.
      - Each bar annotated at its end with the metric value.
    The final image is saved as 'superplot2.png' in the results directory.
    """
    metrics = []
    labels = []
    metric_name = None

    # Load configuration.
    with open(config_path, 'r') as f:
        datasets_config = json.load(f)

    # Iterate over each dataset and mode.
    for dataset_name, config in datasets_config.items():
        data_path = config.get("data_path")
        model_class = config.get("model_class", "XGBClassifier")
        for mode in ["all_data", "not_all_data"]:
            result_folder = os.path.join(results_dir, dataset_name, model_class, mode)
            model_file = os.path.join(result_folder, "model.pkl")
            if not os.path.exists(model_file):
                print(f"Model file not found for {dataset_name} ({mode}). Skipping.")
                continue
            with open(model_file, "rb") as f_model:
                model = pickle.load(f_model)

            # Load and preprocess data.
            df = load_data(data_path)
            if dataset_name == "GDM":
                df = map_gdm_to_binary(df)
            elif dataset_name == "GDM_drugs":
                df = map_gdm_drugs_to_numeric(df)
            elif dataset_name in ["GDM_GA", "out_ga", "BW", "BW_%"]:
                if dataset_name not in df.columns:
                    print(f"Target column '{dataset_name}' not found for {dataset_name} ({mode}). Skipping.")
                    continue
                df = map_others_to_categorical(df, dataset_name)
            elif dataset_name == "PE_PH":
                df = map_peph_to_numeric(df)
            else:
                print(f"Invalid dataset name: {dataset_name}. Skipping.")
                continue

            # Build feature set.
            columns_needed = ID_COLS + BEFORE_COLS + ["y"]
            if mode == "all_data":
                columns_needed += CRL_COLS
            df = df[columns_needed].dropna(subset=["y"]).copy()

            # Split and one-hot encode.
            df_train, df_val, df_test = split_by_id(df)
            for subset in [df_train, df_val, df_test]:
                subset.drop(columns=["id sort", "Scan date"], inplace=True, errors="ignore")
            df_test = one_hot_encode(df_test)
            X_test = df_test.drop(columns=["y"])
            y_test = df_test["y"]

            # Compute performance metric.
            if model_class == "XGBClassifier":
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) > 2:
                    metric_value = roc_auc_score(y_test, y_proba, multi_class="ovr")
                else:
                    y_pred = y_proba[:, 1]
                    metric_value = roc_auc_score(y_test, y_pred)
                metric_name = "AUC"
            elif model_class == "XGBRegressor":
                y_pred = model.predict(X_test)
                metric_value = stats.pearsonr(y_test, y_pred)[0]
                metric_name = "Correlation"
            else:
                continue

            labels.append(f"{dataset_name} ({'with all data' if mode == 'all_data' else 'without all data'})")
            metrics.append(metric_value)

    # Create horizontal bar plot with annotated scores.
    plt.figure(figsize=(10, len(labels) * 0.5 + 2))
    y_pos = np.arange(len(labels))
    bars = plt.barh(y_pos, metrics, align='center', color="skyblue")
    plt.yticks(y_pos, labels)
    plt.xlabel(metric_name)
    plt.title("Evaluation Metrics for XGB")

    # Annotate each bar with its metric value at the end.
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Offset for the text annotation.
        offset = max(metrics) * 0.02 if metrics else 0.1
        plt.text(width + offset, bar.get_y() + bar.get_height() / 2,
                 f"{metrics[i]:.2f}", va="center", ha="left", fontsize=9, color="black")

    plt.tight_layout()
    superplot2_path = os.path.join(results_dir, "superplot2.png")
    plt.savefig(superplot2_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Superplot2 saved to {superplot2_path}")


# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    # First, recreate evaluation plots by loading saved models from each results folder.
    recreate_evaluation_plots("../data/config.json")
    # Then, combine all the evaluation plots into a single superplot.
    generate_superplot("../results")
    # Finally, generate a second superplot with horizontal bar plots for AUC/Correlation.
    generate_superplot2("../results", "../data/config.json")
