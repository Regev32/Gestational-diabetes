#!/usr/bin/env python

"""
GDM_drugs.py

This script loads 'data/gdm_master.csv', converts the "GDM drugs" column into a numeric target
by mapping:
    "diet"             -> 0
    "insulin"          -> 1
    "insulin, metformin" -> 2
    "metformin"        -> 3
(The transformation is case-insensitive and trims whitespace.)
It then selects a feature set that always includes:
    [Age, Race, Height, Weight, BMI 12w, Conception, Smoking, Chronic hypertension, FH DM, Previous GDM,
     Previous FGR, Previous LGA, Last BW%]
plus optionally the CRL-related columns ([CRL, Machine, hCG, PAPP-A, Ut PI]) if all_data is True.
It also retains the "id sort" and "Scan date" columns.

The DataFrame is split by unique "id sort" into 64% train, 16% validation, and 20% test sets.
An XGBoost classifier is trained on the training set and evaluated on the validation set.
The ROC curve (based on the validation set) and the trained model are saved under:
    results/GDM_drugs/all_data/  if all_data is True,
or
    results/GDM_drugs/before_pregnancy/  if all_data is False.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
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


def map_gdm_drugs_to_numeric(df):
    """
    Converts the "GDM drugs" column into a numeric target:
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
    # Clean the column: lowercase and strip
    df["GDM drugs"] = df["GDM drugs"].astype(str).str.lower().str.strip()
    df["GDM drugs"] = df["GDM drugs"].map(mapping)
    df = df[df["GDM drugs"].notna()].copy()
    return df.rename(columns={"GDM drugs": "y"})


def split_by_id(df):
    """
    Splits the DataFrame into 64% train, 16% validation, and 20% test sets,
    grouping by unique "id sort" so that rows for the same ID remain together.
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


def train_and_evaluate_xgb(df_train, df_val):
    """
    Drops "id sort" and "Scan date" from training and validation sets,
    applies one-hot encoding, trains an XGBoost classifier, and computes the ROC curve and AUC on the validation set.
    Returns the trained model, validation AUC, and ROC curve data (fpr, tpr).
    """
    # Make copies to avoid warnings
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

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    y_val_pred_proba = model.predict_proba(X_val)

    # Check number of classes to decide whether to use multi_class parameter
    if len(np.unique(y_val)) > 2:
        val_auc = roc_auc_score(y_val, y_val_pred_proba, multi_class="ovr")
    else:
        val_auc = roc_auc_score(y_val, y_val_pred_proba[:, 1])

    print(f"Validation AUC: {val_auc:.4f}")

    if len(np.unique(y_val)) > 2:
        # For multi-class ROC, we could plot one vs rest ROC for a selected class;
        # here we'll just use the probabilities of the class with highest frequency as an example.
        class_to_plot = y_val.value_counts().idxmax()
        y_val_pred_proba_binary = y_val_pred_proba[:, int(class_to_plot)]
        fpr, tpr, _ = roc_curve((y_val == class_to_plot).astype(int), y_val_pred_proba_binary)
    else:
        fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba[:, 1])

    return model, val_auc, fpr, tpr


def save_results(model, val_auc, fpr, tpr, all_data):
    """
    Saves the ROC curve plot (using the validation set) and the trained model under:
      results/GDM_drugs/all_data/  if all_data is True
      results/GDM_drugs/before_pregnancy/  if all_data is False.
    The ROC plot filename includes the AUC value.
    """
    flag_name = "all_data" if all_data else "before_pregnancy"
    results_dir = os.path.join("results", "GDM_drugs", flag_name)
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {val_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for GDM_drugs Prediction ({flag_name})")
    plt.legend(loc="lower right")
    roc_filename = os.path.join(results_dir, f"gdm_drugs_roc_{val_auc:.4f}.png")
    plt.savefig(roc_filename)
    plt.close()
    print(f"ROC curve saved to {roc_filename}")

    model_filename = os.path.join(results_dir, "xgb_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")


def main(all_data=False, data_path="../data/gdm_4000.csv"):
    df = load_data(data_path)
    df = map_gdm_drugs_to_numeric(df)

    # Build feature set: always include ID_COLS + BEFORE_COLS + ["y"]; include CRL_COLS if all_data is True.
    columns_needed = ID_COLS + BEFORE_COLS + ["y"]
    if all_data:
        columns_needed += CRL_COLS

    df = df[columns_needed].dropna(subset=["y"]).copy()

    df_train, df_val, df_test = split_by_id(df)
    model, val_auc, fpr, tpr = train_and_evaluate_xgb(df_train, df_val)
    save_results(model, val_auc, fpr, tpr, all_data)


if __name__ == "__main__":
    # Set all_data=True to include CRL-related columns; False for only before-pregnancy features.
    main(all_data=False, data_path="../data/gdm_master.csv")
