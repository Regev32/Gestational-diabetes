#!/usr/bin/env python

"""
PEPH.py

This script loads 'data/gdm_master.csv', converts the "PE/PH" column into a numeric target
by mapping:
    "no"    -> 0
    "a pe"  -> 1
    "b pie" -> 2
(The transformation is case-insensitive and ignores extra whitespace.)
It then selects a feature set that always includes:
    [Age, Race, Height, Weight, BMI 12w, Conception, Smoking, Chronic hypertension, FH DM, Previous GDM, Previous FGR, Previous LGA, Last BW%]
plus optionally the CRL-related columns ([CRL, Machine, hCG, PAPP-A, Ut PI]) if all_data is True.
It also retains the "id sort" and "Scan date" columns.

The data is then split by unique "id sort" into 64% train, 16% validation, and 20% test sets.
An XGBoost model is trained on the training set and evaluated on the validation set.
The ROC curve (based on the validation set) and the trained model are saved under:
    results/PEPH/all_data/   (if all_data=True)
or
    results/PEPH/before_pregnancy/   (if all_data=False)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Define column groups
BEFORE_COLS = [
    "Age", "Race", "Height", "Weight", "BMI 12w", "Conception", "Smoking",
    "Chronic hypertension", "FH DM", "Previous GDM", "Previous FGR", "Previous LGA", "Last BW%"
]
CRL_COLS = ["CRL", "Machine", "hCG", "PAPP-A", "Ut PI"]
ID_COLS = ["id sort", "Scan date"]

def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

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
    df["PE/PH"] = df["PE/PH"].astype(str).str.lower().str.strip().map(mapping)
    df = df[df["PE/PH"].notna()].copy()
    return df.rename(columns={"PE/PH": "y"})

def split_by_id(df):
    """
    Splits the DataFrame into 64% train, 16% validation, and 20% test sets,
    grouping by unique "id sort" so that rows from the same ID remain together.
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
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    return df_encoded

def tune_xgb_hyperparams(X_train, y_train):
    """
    Uses RandomizedSearchCV to find good hyperparameters for XGBClassifier
    based on ROC AUC. Returns the best estimator found.
    """
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 2, 5, 10]
    }
    xgb_clf = XGBClassifier(eval_metric="logloss", random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print("Best params found:", random_search.best_params_)
    print("Best CV AUC:", random_search.best_score_)
    return random_search.best_estimator_

def train_and_evaluate_xgb(df_train, df_val, hyperparam_tuning=True):
    """
    Drops "id sort" and "Scan date" from training and validation sets,
    applies one-hot encoding, trains an XGBoost classifier (with optional hyperparameter tuning),
    and computes the ROC curve and AUC on the validation set.
    Returns the trained model, validation AUC, and ROC curve data (fpr, tpr).
    """
    # Work on copies to avoid warnings
    df_train = df_train.copy()
    df_val = df_val.copy()
    for subset in [df_train, df_val]:
        subset.drop(columns=["id sort", "Scan date"], inplace=True, errors="ignore")

    df_train = one_hot_encode(df_train)
    df_val = one_hot_encode(df_val)

    X_train, y_train = df_train.drop(columns=["y"]), df_train["y"]
    X_val, y_val = df_val.drop(columns=["y"]), df_val["y"]

    print("Training set class distribution:")
    print(y_train.value_counts())
    print("Validation set class distribution:")
    print(y_val.value_counts())

    if hyperparam_tuning:
        print("Starting hyperparameter tuning for XGBClassifier...")
        model = tune_xgb_hyperparams(X_train, y_train)
    else:
        print("Using default XGBClassifier parameters...")
        model = XGBClassifier(eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)

    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Validation AUC: {val_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
    return model, val_auc, fpr, tpr

def save_results(model, val_auc, fpr, tpr, all_data):
    """
    Saves the ROC curve plot (using the validation set) and the trained model under:
      results/PEPH/all_data/   if all_data is True
      results/PEPH/before_pregnancy/   if all_data is False
    The ROC plot filename includes the AUC value.
    """
    flag_name = "all_data" if all_data else "before_pregnancy"
    results_dir = os.path.join("results", "PEPH", flag_name)
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {val_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for PE/PH Prediction ({flag_name})")
    plt.legend(loc="lower right")
    roc_filename = os.path.join(results_dir, f"peph_roc_{val_auc:.4f}.png")
    plt.savefig(roc_filename)
    plt.close()
    print(f"ROC curve saved to {roc_filename}")

    model_filename = os.path.join(results_dir, "xgb_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")

def main(all_data=False, data_path="../data/gdm_master.csv", hyperparam_tuning=True):
    # Load the data
    df = load_data(data_path)
    # Map PE/PH to numeric values (dropping rows where mapping fails)
    df = map_peph_to_numeric(df)

    # Build feature set: always include ID_COLS + BEFORE_COLS + ["y"]
    # If all_data is True, also include CRL_COLS.
    columns_needed = ID_COLS + BEFORE_COLS + ["y"]
    if all_data:
        columns_needed += CRL_COLS

    df = df[columns_needed].dropna(subset=["y"]).copy()

    df_train, df_val, df_test = split_by_id(df)
    model, val_auc, fpr, tpr = train_and_evaluate_xgb(df_train, df_val, hyperparam_tuning=hyperparam_tuning)
    save_results(model, val_auc, fpr, tpr, all_data)

if __name__ == "__main__":
    # Set all_data=True to include CRL-related columns, or False to use only before-pregnancy features.
    # Toggle hyperparam_tuning=True to enable hyperparameter tuning.
    main(all_data=False, data_path="../data/gdm_master.csv", hyperparam_tuning=True)
