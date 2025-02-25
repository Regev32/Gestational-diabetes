#!/usr/bin/env python

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


def map_gdm_to_binary(df):
    mapping = {"no": 0, "a gdm": 1}
    df["GDM"] = df["GDM"].astype(str).str.lower().str.strip().map(mapping)
    df = df[df["GDM"].notna()].copy()
    return df.rename(columns={"GDM": "y"})


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
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    return df_encoded


def train_and_evaluate_xgb(df_train, df_val):
    # Drop id columns in each subset (make a copy to avoid SettingWithCopyWarning)
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

    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Validation AUC: {val_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
    return model, val_auc, fpr, tpr


def save_results(model, val_auc, fpr, tpr, all_data):
    """
    Saves the ROC curve plot and the trained model under "results/GDM/all_data/".
    The ROC plot filename includes the AUC.
    """
    flag_name = "all_data" if all_data else "before_pregnancy"
    results_dir = os.path.join("results", "GDM", flag_name)
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {val_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for GDM Prediction ({flag_name})")
    plt.legend(loc="lower right")
    roc_filename = os.path.join(results_dir, f"gdm_roc_{val_auc:.4f}.png")
    plt.savefig(roc_filename)
    plt.close()
    print(f"ROC curve saved to {roc_filename}")

    model_filename = os.path.join(results_dir, "xgb_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")


def main(all_data=False, data_path="../data/gdm_master.csv"):
    df = load_data(data_path)
    df = map_gdm_to_binary(df)

    # Build feature set: always include ID_COLS + BEFORE_COLS + ["y"]; include CRL_COLS if flag is set.
    columns_needed = ID_COLS + BEFORE_COLS + ["y"]
    if all_data:
        columns_needed += CRL_COLS

    df = df[columns_needed].dropna(subset=["y"]).copy()

    df_train, df_val, df_test = split_by_id(df)
    model, val_auc, fpr, tpr = train_and_evaluate_xgb(df_train, df_val)
    save_results(model, val_auc, fpr, tpr, all_data)


if __name__ == "__main__":
    # Set `all_data=True` for including CRL columns, `False` otherwise
    main(all_data=False, data_path="../data/gdm_master.csv")  # Change to False if needed
