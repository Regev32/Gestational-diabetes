#!/usr/bin/env python

"""
GDM_GA.py

This script loads 'data/gdm_master.csv' and trains a regression model to predict a continuous target outcome,
specifically for "GDM GA". The script always uses the following feature groups:
  - ID columns: ["id sort", "Scan date"]
  - BEFORE-pregnancy features: [Age, Race, Height, Weight, BMI 12w, Conception, Smoking,
      Chronic hypertension, FH DM, Previous GDM, Previous FGR, Previous LGA, Last BW%]
Optionally (if all_data is True) it also includes the CRL-related columns:
  [CRL, Machine, hCG, PAPP-A, Ut PI]

The data is split (by unique "id sort") into 64% train, 16% validation, and 20% test sets.
All remaining categorical columns (other than "id sort", "Scan date", and "y") are one-hot encoded.
An XGBoost regressor is trained on the training set and evaluated on the validation set using RMSE (and R²).
A scatter plot of predicted vs. actual values is generated and saved, and the model is saved as a pickle file.
The results are saved under:
    results/GDM_GA/all_data/   if all_data is True,
or
    results/GDM_GA/before_pregnancy/   if all_data is False.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Define feature groups
BEFORE_COLS = [
    "Age", "Race", "Height", "Weight", "BMI 12w", "Conception", "Smoking",
    "Chronic hypertension", "FH DM", "Previous GDM", "Previous FGR", "Previous LGA", "Last BW%"
]
CRL_COLS = ["CRL", "Machine", "hCG", "PAPP-A", "Ut PI"]
ID_COLS = ["id sort", "Scan date"]


def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)


def prepare_data(df, target_col, all_data):
    """
    Prepares the data for regression:
      - Selects columns: ID_COLS + BEFORE_COLS + [target_col] and, if all_data is True, CRL_COLS.
      - Drops rows with missing values in the target.
      - Renames the target column to "y".
    """
    columns_needed = ID_COLS + BEFORE_COLS + [target_col]
    if all_data:
        columns_needed += CRL_COLS
    df = df[columns_needed].dropna(subset=[target_col]).copy()
    return df.rename(columns={target_col: "y"})


def split_by_id(df):
    """
    Splits the DataFrame by unique "id sort" into 64% train, 16% validation, and 20% test sets.
    """
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
    """
    One-hot encodes all object-type columns except for "id sort", "Scan date", and "y".
    """
    exclude_cols = {"id sort", "Scan date", "y"}
    cat_cols = [col for col in df.select_dtypes(include=["object"]).columns if col not in exclude_cols]
    if cat_cols:
        return pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        return df.copy()


def tune_xgb_regressor_hyperparams(X_train, y_train):
    """
    Uses RandomizedSearchCV to find good hyperparameters for XGBRegressor
    based on negative root mean squared error.
    Returns the best estimator found.
    """
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgb_reg = XGBRegressor(eval_metric="rmse", random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print("Best params found by RandomizedSearchCV:", random_search.best_params_)
    print("Best CV RMSE from RandomizedSearchCV:", -random_search.best_score_)
    return random_search.best_estimator_


def train_and_evaluate_xgb(df_train, df_val, hyperparam_tuning=True):
    """
    Drops "id sort" and "Scan date", applies one-hot encoding, trains an XGBoost regressor (with optional hyperparameter tuning),
    and computes RMSE and R² on the validation set.
    Returns the trained model, RMSE, y_val, and predicted y_val.
    """
    # Work on copies to avoid warnings
    df_train = df_train.copy()
    df_val = df_val.copy()
    for subset in [df_train, df_val]:
        subset.drop(columns=["id sort", "Scan date"], inplace=True, errors="ignore")
    df_train = one_hot_encode(df_train)
    df_val = one_hot_encode(df_val)

    # Align validation set columns to training set
    df_val = df_val.reindex(columns=df_train.columns, fill_value=0)

    X_train, y_train = df_train.drop(columns=["y"]), df_train["y"]
    X_val, y_val = df_val.drop(columns=["y"]), df_val["y"]

    if hyperparam_tuning:
        print("Starting hyperparameter tuning for XGBRegressor...")
        model = tune_xgb_regressor_hyperparams(X_train, y_train)
    else:
        print("Using default XGBRegressor parameters...")
        model = XGBRegressor(eval_metric="rmse", random_state=42)
        model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R^2: {r2:.4f}")

    return model, rmse, y_val, y_val_pred


def save_results(model, rmse, y_val, y_val_pred, target_col, all_data):
    """
    Saves a scatter plot of predicted vs. actual values and the trained model under:
      results/<target_sanitized>/<flag_name>/
    where <target_sanitized> is target_col with spaces/slashes replaced,
    and <flag_name> is "all_data" if all_data is True or "before_pregnancy" otherwise.
    The plot filename includes the RMSE.
    """
    target_folder_name = target_col.replace(" ", "_").replace("/", "_")
    flag_name = "all_data" if all_data else "before_pregnancy"
    results_dir = os.path.join("results", target_folder_name, flag_name)
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicted vs. Actual for {target_col} (RMSE = {rmse:.4f})")
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
    plot_filename = os.path.join(results_dir, f"{target_folder_name}_pred_vs_actual_RMSE_{rmse:.4f}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Prediction plot saved to {plot_filename}")

    model_filename = os.path.join(results_dir, "xgb_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")


def main(all_data=False, data_path="../data/gdm_4000.csv", target_col="GDM GA", hyperparam_tuning=True):
    df = load_data(data_path)
    df = prepare_data(df, target_col, all_data)
    df_train, df_val, df_test = split_by_id(df)
    model, rmse, y_val, y_val_pred = train_and_evaluate_xgb(df_train, df_val, hyperparam_tuning=hyperparam_tuning)
    save_results(model, rmse, y_val, y_val_pred, target_col, all_data)


if __name__ == "__main__":
    # Example usage: predicting "GDM GA" as a continuous outcome.
    # Set all_data=True to include CRL-related columns; adjust hyperparam_tuning as desired.
    main(all_data=True, data_path="../data/gdm_4000.csv", target_col="GDM GA", hyperparam_tuning=True)
