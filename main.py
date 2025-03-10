#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import warnings

import optuna
import optuna.visualization.matplotlib as matviz
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import ExperimentalWarning

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, roc_curve

def load_model(model_class, hypers_values):
    if model_class == 'XGBClassifier':
        return XGBClassifier(**hypers_values)
    elif model_class == 'XGBRegressor':
        return XGBRegressor(**hypers_values)
    else:
        raise ValueError(f"Unsupported model '{model_class}'")

# -------------------------------------------------------------------
# Settings & Constants
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ExperimentalWarning)
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
# Define column groups
BEFORE_COLS = [
    "Age", "Race", "Height", "Weight", "BMI 12w", "Conception", "Smoking",
    "Chronic hypertension", "FH DM", "Previous GDM", "Previous FGR", "Previous LGA", "Last BW%"
]
CRL_COLS = ["CRL", "Machine", "hCG", "PAPP-A", "Ut PI"]
ID_COLS = ["id sort", "Scan date"]

# -------------------------------------------------------------------
# Data Handling Functions
# -------------------------------------------------------------------
def load_data(file_path):
    """Load CSV data from the provided file path."""
    return pd.read_csv(file_path, low_memory=False)

def map_gdm_to_binary(df):
    """Map GDM column to binary and rename to 'y'."""
    mapping = {"no": 0, "a gdm": 1}
    df["GDM"] = df["GDM"].astype(str).str.lower().str.strip().map(mapping)
    df = df[df["GDM"].notna()].copy()
    return df.rename(columns={"GDM": "y"})

def map_gdm_drugs_to_numeric(df):
    """
    Converts the "GDM_drugs" column into a numeric target:
      "diet"             -> 0
      "insulin"          -> 1
      "insulin, metformin" -> 2
      "metformin"        -> 3
    Unmapped rows are dropped. The column is then renamed to "y".
    """
    mapping = {
        "diet": 0,
        "insulin": 1,
        "insulin, metformin": 2,
        "metformin": 3
    }
    df["GDM_drugs"] = df["GDM_drugs"].astype(str).str.lower().str.strip()
    df["GDM_drugs"] = df["GDM_drugs"].map(mapping)
    df = df[df["GDM_drugs"].notna()].copy()
    return df.rename(columns={"GDM_drugs": "y"})

def split_by_id(df):
    """Split dataframe into train, validation, and test sets based on unique IDs."""
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
    """One-hot encode categorical columns, excluding specified ones."""
    exclude_cols = {"id sort", "Scan date", "y"}
    cat_cols = [col for col in df.select_dtypes(include=["object"]).columns if col not in exclude_cols]
    if cat_cols:
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    return df_encoded

def map_others_to_categorical(df, target_col):
    """
    Prepares the data for regression by:
      - Dropping rows with missing values in the target.
      - Renaming the target column to "y".
    """
    df = df.dropna(subset=[target_col]).copy()
    return df.rename(columns={target_col: "y"})

def map_peph_to_numeric(df):
    """
    Converts the PE/PH column into a numeric target:
      "no"    -> 0
      "a pe"  -> 1
      "b pie" -> 2
    Rows with unmapped values are dropped.
    The column is then renamed to "y".
    """
    mapping = {"no": 0, "a pe": 1, "b pie": 2}
    df["PE_PH"] = df["PE_PH"].astype(str).str.lower().str.strip().map(mapping)
    df = df[df["PE_PH"].notna()].copy()
    return df.rename(columns={"PE_PH": "y"})

def plot_and_save_roc_curve(y_true, y_pred, save_filepath):
    """Plot the ROC curve and save the figure."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()

# New helper for regression: scatter plot of predicted vs. actual
def plot_regression_results(y_true, y_pred, save_filepath):
    """
    Plot a scatter plot of actual vs. predicted values for regression,
    with a reference line (y = x) to indicate perfect predictions.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------
# Model Optimization Functions
# -------------------------------------------------------------------
def objective(trial, X_train, y_train, X_val, y_val, hyperparameters_dict, model_class):
    """
    Objective function for hyperparameter optimization.
    """
    hypers_values = {}
    for hyper_name, hyper_spec in hyperparameters_dict.items():
        type_ = hyper_spec['type']
        distribution = hyper_spec.get('distribution', None)
        should_log = distribution == 'loguniform'

        if type_ == 'int':
            hypers_values[hyper_name] = trial.suggest_int(
                hyper_name, int(hyper_spec['min']), int(hyper_spec['max']), log=should_log
            )
        elif type_ == 'float':
            hypers_values[hyper_name] = trial.suggest_float(
                hyper_name, hyper_spec['min'], hyper_spec['max'], log=should_log
            )
        elif type_ == 'categorical':
            hypers_values[hyper_name] = trial.suggest_categorical(
                hyper_name, hyper_spec['values']
            )
        else:
            raise ValueError(f"Unsupported type '{type_}' for hyperparameter '{hyper_name}'")

    model = load_model(model_class, hypers_values)
    model.fit(X_train, y_train)

    if model_class == "XGBClassifier":
        y_val_pred = model.predict_proba(X_val)
        if len(np.unique(y_val)) > 2:
            score = roc_auc_score(y_val, y_val_pred, multi_class="ovr")
        else:
            score = roc_auc_score(y_val, y_val_pred[:, 1])
    elif model_class == "XGBRegressor":
        y_val_pred = model.predict(X_val)
        from sklearn.metrics import mean_squared_error
        score = -mean_squared_error(y_val, y_val_pred)
    else:
        raise ValueError(f"Unsupported model '{model_class}'")

    return score

def run_optimization(X_train, y_train, X_val, y_val, hyperparameters_dict, n_trials=100, model_class="XGBClassifier"):
    """Run hyperparameter optimization and return the study."""
    objective_func = lambda trial: objective(trial, X_train, y_train, X_val, y_val, hyperparameters_dict, model_class)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner(n_warmup_steps=10))
    study.optimize(objective_func, n_trials=n_trials)
    return study

# -------------------------------------------------------------------
# Plotting & Saving Functions
# -------------------------------------------------------------------
def ensure_dir_exists(path):
    """Ensure that the directory exists."""
    os.makedirs(path, exist_ok=True)

def save_matplotlib_plot(plot_func, save_filepath, *args, **kwargs):
    """
    Call the provided plotting function, then save the current figure.
    """
    plot_func(*args, **kwargs)
    fig = plt.gcf()
    fig.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_auc_trials(auc_values, save_filepath):
    """Plot the Score vs. trial number and save the figure."""
    plt.figure()
    plt.plot(range(1, len(auc_values) + 1), auc_values, marker='o')
    plt.xlabel("Trial")
    plt.ylabel("Score")
    plt.title("Score vs. Trials")
    plt.grid()
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------
# Main Workflow
# -------------------------------------------------------------------
def main(all_data=False, data_path="data/gdm_master.csv", dataset_name="GDM", model_class="XGBClassifier"):
    # Determine subfolder based on the all_data flag
    data_flag = "all_data" if all_data else "not_all_data"
    BASE_SAVE_DIR = "results"
    MODEL_SAVE_DIR = os.path.join(BASE_SAVE_DIR, dataset_name, model_class, data_flag)

    # Define hyperparameter search space for XGBoost
    hyperparameters = xgb_hyperparameters

    # Load and preprocess the data
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
        print("Must give valid model name!")
        return

    # Build feature set (include CRL_COLS if all_data flag is set)
    columns_needed = ID_COLS + BEFORE_COLS + ["y"]
    if all_data:
        columns_needed += CRL_COLS
    df = df[columns_needed].dropna(subset=["y"]).copy()

    # Split into train, validation, and test sets
    df_train, df_val, df_test = split_by_id(df)
    for subset in [df_train, df_val, df_test]:
        subset.drop(columns=["id sort", "Scan date"], inplace=True, errors="ignore")
    df_train = one_hot_encode(df_train)
    df_val = one_hot_encode(df_val)
    df_test = one_hot_encode(df_test)

    X_train, y_train = df_train.drop(columns=["y"]), df_train["y"]
    X_val, y_val = df_val.drop(columns=["y"]), df_val["y"]
    X_test, y_test = df_test.drop(columns=["y"]), df_test["y"]

    # Run hyperparameter optimization
    study = run_optimization(X_train, y_train, X_val, y_val, hyperparameters_dict=hyperparameters, n_trials=10000, model_class=model_class)
    best_params = study.best_params

    ensure_dir_exists(MODEL_SAVE_DIR)
    with open(os.path.join(MODEL_SAVE_DIR, "best_params.json"), 'w') as f:
        json.dump(best_params, f)

    # Generate and save plots using Optuna's matplotlib visualization
    params_names = list(best_params.keys())
    slice_plot_path = os.path.join(MODEL_SAVE_DIR, "slice_plot.png")
    save_matplotlib_plot(matviz.plot_slice, slice_plot_path, study, params=params_names)
    importance_plot_path = os.path.join(MODEL_SAVE_DIR, "param_importance_plot.png")
    save_matplotlib_plot(matviz.plot_param_importances, importance_plot_path, study)

    # Train final model using best hyperparameters and save it
    model = load_model(model_class, best_params)
    model.fit(X_train, y_train)
    with open(os.path.join(MODEL_SAVE_DIR, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)

    # Evaluation: Different metrics/plots for classifier vs. regressor
    if model_class == "XGBClassifier":
        y_test_proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) > 2:
            test_metric = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
            print(f"Test AUC (multi-class) = {test_metric:.4f} for dataset '{dataset_name}' with {'all data' if all_data else 'not all data'}")
        else:
            y_test_pred = y_test_proba[:, 1]
            test_metric = roc_auc_score(y_test, y_test_pred)
            print(f"Test AUC = {test_metric:.4f} for dataset '{dataset_name}' with {'all data' if all_data else 'not all data'}")
            roc_curve_path = os.path.join(MODEL_SAVE_DIR, "roc_curve.png")
            plot_and_save_roc_curve(y_test, y_test_pred, roc_curve_path)
    elif model_class == "XGBRegressor":
        y_test_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        test_metric = mean_squared_error(y_test, y_test_pred)
        print(f"Test MSE = {test_metric:.4f} for dataset '{dataset_name}' with {'all data' if all_data else 'not all data'}")
        # Save a regression scatter plot (predicted vs. actual)
        regression_plot_path = os.path.join(MODEL_SAVE_DIR, "regression_scatter.png")
        plot_regression_results(y_test, y_test_pred, regression_plot_path)
    else:
        raise ValueError(f"Unsupported model '{model_class}'")

    auc_values = [trial.value for trial in study.trials]
    auc_plot_path = os.path.join(MODEL_SAVE_DIR, "auc_vs_trials.png")
    plot_auc_trials(auc_values, auc_plot_path)
    plt.show()

    print(f"Best validation value = {max(auc_values):.4f} for dataset '{dataset_name}' with {'all data' if all_data else 'not all data'}")

# -------------------------------------------------------------------
# Entry Point: Load configuration from "data/config.json" and loop
# -------------------------------------------------------------------
if __name__ == "__main__":
    config_path = os.path.join("data", "config.json")
    with open(config_path, 'r') as f:
        datasets_config = json.load(f)

    for dataset_name, config in datasets_config.items():
        print(f"\nProcessing dataset: {dataset_name}")
        data_path = config.get("data_path")
        model_class = config.get("model_class", "XGBClassifier")
        for all_data in [True, False]:
            main(all_data=all_data, data_path=data_path, dataset_name=dataset_name, model_class=model_class)
