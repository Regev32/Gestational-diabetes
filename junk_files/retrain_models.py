#!/usr/bin/env python
"""
This script retrains all models in case of an accident. It loads the best hyperparameters from
the saved best_params.json file (assumed to exist in the appropriate results folder), retrains
the model on the training data, and then generates:
  - A hyperparameter importance plot (a horizontal bar chart showing best hyperparameters,
    with each bar annotated with its value at the end).
  - An evaluation plot:
      * For XGBClassifier: a ROC curve (binary or one-vs-rest for multi-class).
      * For XGBRegressor: a regression scatter plot (actual vs. predicted).
  - A feature importance plot (if available).

Before saving the model and each plot, any existing file at that location is deleted.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------
# Settings & Constants
# ---------------------------
# Best hyperparameters were already found using Optuna previously.
# The following dictionary is used for reference of hyperparameter names and expected types.
xgb_hyperparameters = {
    "n_estimators": {"type": "int"},
    "max_depth": {"type": "int"},
    "min_child_weight": {"type": "float"},
    "subsample": {"type": "float"},
    "learning_rate": {"type": "float"},
    "colsample_bylevel": {"type": "float"},
    "colsample_bytree": {"type": "float"},
    "gamma": {"type": "float"},
    "lambda": {"type": "float"},
    "alpha": {"type": "float"}
}

# Column groups for feature selection:
BEFORE_COLS = [
    "Age", "Race", "Height", "Weight", "BMI 12w", "Conception", "Smoking",
    "Chronic hypertension", "Previous GDM", "Previous FGR", "Previous LGA", "Last BW%"
]
CRL_COLS = ["CRL", "Machine", "hCG", "PAPP-A", "Ut PI"]
ID_COLS = ["id sort", "Scan date"]


# ---------------------------
# Helper Functions
# ---------------------------
def delete_if_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

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

def load_model(model_class, hypers_values):
    from xgboost import XGBClassifier, XGBRegressor
    if model_class == "XGBClassifier":
        return XGBClassifier(**hypers_values)
    elif model_class == "XGBRegressor":
        return XGBRegressor(**hypers_values)
    else:
        raise ValueError(f"Unsupported model '{model_class}'")

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
    rho = stats.pearsonr(y_true, y_pred)[0]
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.text(0.05, 0.95, f"Correlation = {rho:.2f}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.5))
    plt.legend()
    delete_if_exists(save_filepath)
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, features, title, save_filepath):
    """
    Creates a horizontal bar chart of feature importances, aggregating one-hot
    features by their prefix. Values are displayed on the x-axis, and the feature
    groups on the y-axis.
    """
    importance = model.feature_importances_
    agg_importance = {}
    for feat, imp in zip(features, importance):
        if "_" in feat:
            prefix = feat.split("_")[0]
            agg_importance[prefix] = agg_importance.get(prefix, 0) + imp
        else:
            agg_importance[feat] = imp
    sorted_items = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)
    agg_features = [item[0] for item in sorted_items]
    agg_values = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    y_positions = range(len(agg_values))
    bars = plt.barh(y_positions, agg_values, align='center', color='skyblue')
    plt.yticks(y_positions, agg_features)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()

    # Remove old file if it exists.
    delete_if_exists(save_filepath)

    # Optionally annotate each bar with its importance (if you want).
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                 f"{agg_values[i]:.2f}", va="center", ha="left", fontsize=8)

    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# Retraining Workflow
# ---------------------------
def retrain_models(config_path="data/config.json"):
    """
    For each dataset defined in the configuration file and for both modes
    ("all_data" and "not_all_data"), this function:
      - Loads and preprocesses the CSV data.
      - Splits the data and applies one-hot encoding.
      - Loads the best hyperparameters from the saved best_params.json file.
      - Retrains the model using those hyperparameters.
      - Deletes any existing saved model and plot files.
      - Saves the retrained model.
      - Generates and saves:
            * A hyperparameter importance plot (bar chart of best hyperparameters, with scores annotated),
            * An evaluation plot (ROC curve for classifier or regression scatter for regressor),
            * A feature importance plot (if available).
    """
    with open(config_path, 'r') as f:
        datasets_config = json.load(f)

    for dataset_name, config in datasets_config.items():
        data_path = config.get("data_path")
        model_class = config.get("model_class", "XGBClassifier")
        # Loop over both modes: all_data True/False
        for all_data in [True, False]:
            data_flag = "all_data" if all_data else "not_all_data"
            # Define results folder where best_params.json is stored
            results_folder = os.path.join("../results", dataset_name, model_class, data_flag)
            print(f"\nRetraining model for {dataset_name} ({model_class}, {data_flag}) in folder {results_folder}")

            # Build file paths for saved files
            best_params_file = os.path.join(results_folder, "best_params.json")
            model_file = os.path.join(results_folder, "model.pkl")
            hyperparam_plot_file = os.path.join(results_folder, "hyperparameter_importance_plot.png")
            eval_plot_file = os.path.join(
                results_folder,
                "roc_curve.png" if model_class == "XGBClassifier" else "regression_scatter.png"
            )
            feature_imp_plot_file = os.path.join(results_folder, "feature_importance_plot.png")

            # Check that best_params_file exists
            if not os.path.exists(best_params_file):
                print(f"Best parameters file not found: {best_params_file}. Skipping {dataset_name} {data_flag}.")
                continue

            # Load best hyperparameters
            with open(best_params_file, 'r') as bp_file:
                best_params = json.load(bp_file)

            # Load and preprocess data
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
                print(f"Invalid dataset name: {dataset_name}. Skipping.")
                continue

            # Build feature set
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
            X_train, y_train = df_train.drop(columns=["y"]), df_train["y"]
            X_test, y_test = df_test.drop(columns=["y"]), df_test["y"]

            # Retrain model
            model = load_model(model_class, best_params)
            model.fit(X_train, y_train)
            delete_if_exists(model_file)
            with open(model_file, 'wb') as f_model:
                pickle.dump(model, f_model)
            print(f"Saved retrained model to {model_file}")

            data_text = "with all data" if all_data else "without all data"

            # Evaluation and evaluation plot
            if model_class == "XGBClassifier":
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) > 2:
                    # Multi-class: pick majority class as positive
                    majority_class = y_test.value_counts().idxmax()
                    y_test_binary = (y_test == majority_class).astype(int)
                    y_proba_binary = y_proba[:, int(majority_class)]
                    plot_and_save_roc_curve(y_test_binary, y_proba_binary, eval_plot_file)
                    print(f"Saved one-vs-rest ROC curve for class '{majority_class}' to {eval_plot_file}")
                else:
                    y_pred = y_proba[:, 1]
                    plot_and_save_roc_curve(y_test, y_pred, eval_plot_file)
                    print(f"Saved ROC curve to {eval_plot_file}")
            elif model_class == "XGBRegressor":
                y_pred = model.predict(X_test)
                plot_regression_results(y_test, y_pred, eval_plot_file)
                print(f"Saved regression scatter plot to {eval_plot_file}")
            else:
                print(f"Unsupported model: {model_class}")

            # Generate feature importance plot if available
            if hasattr(model, "feature_importances_"):
                feature_title = f"Feature Importance for {dataset_name} {data_text}"
                plot_feature_importance(model, np.array(X_train.columns), feature_title, feature_imp_plot_file)
                print(f"Saved feature importance plot to {feature_imp_plot_file}")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    # Retrain models using the configuration from data/config.json
    retrain_models("../data/config.json")
