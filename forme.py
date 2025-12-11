"""
Train regression models to predict the eyes location from other keypoints
using train_on_all as the training set, and evaluate also on an external
Roboflow-style dataset: prawn_2025_circ_small_v1.

Keypoint index mapping (Roboflow):
    0 = carapace-start
    1 = eyes
    2 = rostrum
    3 = tail
"""

import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\carmonta\Desktop\ML_Course_project"

TRAIN_ROOT = os.path.join(PROJECT_ROOT, "train_on_all")
TEST_ROOT  = os.path.join(PROJECT_ROOT, "prawn_2025_circ_small_v1")

# לכל דאטהסט יש תיקיות images / labels
TRAIN_LABELS_DIR = os.path.join(TRAIN_ROOT, "labels")
TEST_LABELS_DIR  = os.path.join(TEST_ROOT, "labels")


# ============================================================
# HELPERS
# ============================================================
def parse_labels_dir(labels_dir: str) -> pd.DataFrame:
    """
    Parse YOLO keypoint label .txt files from a labels directory into
    a long-format DataFrame with columns:
        image_stem, object_id, keypoint_index, x_norm, y_norm, visibility
    This version searches RECURSIVELY (labels/train, labels/valid, labels/test, etc.)
    """
    rows = []

    # *** שינוי כאן: חיפוש רקורסיבי בכל התתי-תקיות ***
    pattern = os.path.join(labels_dir, "**", "*.txt")
    label_files = glob.glob(pattern, recursive=True)

    print(f"Searching for labels in (recursive): {labels_dir}")
    print(f"Found {len(label_files)} label files")

    for label_path in label_files:
        image_stem = os.path.splitext(os.path.basename(label_path))[0]

        with open(label_path, "r") as f:
            for obj_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) <= 5:
                    # no keypoints
                    continue

                cls = int(float(parts[0]))
                nums = list(map(float, parts[1:]))

                # first 4 = bbox
                cx, cy, bw, bh = nums[:4]
                kpts = nums[4:]

                if len(kpts) % 3 != 0:
                    print(f"Warning: keypoints len not divisible by 3 in {label_path}")
                    continue

                n_kpts = len(kpts) // 3
                for k_idx in range(n_kpts):
                    x = kpts[3 * k_idx + 0]
                    y = kpts[3 * k_idx + 1]
                    v = kpts[3 * k_idx + 2]

                    rows.append({
                        "image_stem": image_stem,
                        "object_id": obj_id,
                        "class_id": cls,
                        "keypoint_index": k_idx,
                        "x_norm": x,
                        "y_norm": y,
                        "visibility": v,
                    })

    df = pd.DataFrame(rows)
    print(f"Total keypoints parsed from {labels_dir}: {len(df)}")
    return df



def make_wide(df_long: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Convert long-format keypoints DataFrame to wide format:
        one row per (image_stem, object_id)
        columns: carapace_x, carapace_y, eyes_x, eyes_y, rostrum_x, rostrum_y, tail_x, tail_y
    according to mapping:
        0 = carapace-start, 1 = eyes, 2 = rostrum, 3 = tail
    """
    if df_long.empty:
        raise ValueError(f"{dataset_name}: no rows in df_long")

    # להשתמש רק ב-visibility=2 (נקודות טובות)
    df_visible = df_long[df_long["visibility"] == 2].copy()

    # pivot: שורה לכל (image_stem, object_id)
    df_wide = df_visible.pivot_table(
        index=["image_stem", "object_id"],
        columns="keypoint_index",
        values=["x_norm", "y_norm"]
    )

    # ('x_norm', 0) -> 'x_norm0'
    df_wide.columns = [f"{coord}{kpt_idx}" for (coord, kpt_idx) in df_wide.columns]
    df_wide = df_wide.reset_index()

    # מיפוי לפי מה שהגדרת:
    # 0 = carapace-start
    # 1 = eyes
    # 2 = rostrum
    # 3 = tail
    rename_map = {
        "x_norm0": "carapace_x",
        "y_norm0": "carapace_y",
        "x_norm1": "eyes_x",
        "y_norm1": "eyes_y",
        "x_norm2": "rostrum_x",
        "y_norm2": "rostrum_y",
        "x_norm3": "tail_x",
        "y_norm3": "tail_y",
    }
    df_wide = df_wide.rename(columns=rename_map)

    required_cols = [
        "carapace_x", "carapace_y",
        "eyes_x", "eyes_y",
        "rostrum_x", "rostrum_y",
        "tail_x", "tail_y",
    ]
    before = len(df_wide)
    df_wide = df_wide.dropna(subset=required_cols)
    after = len(df_wide)

    print(f"{dataset_name}: prawns with full keypoints: {after}/{before}")

    return df_wide


def train_on_trainset(df_train: pd.DataFrame):
    """
    Train models on train set (with internal train/test split),
    and return the trained models + internal test metrics.
    """
    feature_cols = [
        "carapace_x", "carapace_y",
        "rostrum_x", "rostrum_y",
        "tail_x", "tail_y",
    ]
    target_cols = ["eyes_x", "eyes_y"]

    X = df_train[feature_cols].values
    y = df_train[target_cols].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Internal split -> Train size:", len(X_tr), " | Test size:", len(X_te))

    # baseline: mean eyes
    mean_eyes = y_tr.mean(axis=0)
    y_pred_base = np.tile(mean_eyes, (len(y_te), 1))
    mae_base = mean_absolute_error(y_te, y_pred_base, multioutput="raw_values")

    models = {
        "linear_regression": LinearRegression(),
        "knn_k5": KNeighborsRegressor(n_neighbors=5),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "mlp_small": MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=500,
            random_state=42
        ),
    }

    internal_results = {"baseline_mean": mae_base}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae_xy = mean_absolute_error(y_te, y_pred, multioutput="raw_values")
        internal_results[name] = mae_xy

    print("\n=========== INTERNAL TEST RESULTS (normalized MAE) ===========")
    print("Model              | MAE_x      | MAE_y")
    print("-------------------+------------+-----------")
    for name, mae_xy in internal_results.items():
        print(f"{name:18s} | {mae_xy[0]:10.4f} | {mae_xy[1]:10.4f}")

    return models, feature_cols, target_cols


def evaluate_on_external(df_test: pd.DataFrame, models, feature_cols, target_cols):
    """
    Evaluate the trained models on an external dataset (no splitting, full set as test).
    """
    X_ext = df_test[feature_cols].values
    y_ext = df_test[target_cols].values

    # baseline
    mean_eyes_ext = y_ext.mean(axis=0)
    y_pred_base = np.tile(mean_eyes_ext, (len(y_ext), 1))
    mae_base = mean_absolute_error(y_ext, y_pred_base, multioutput="raw_values")

    external_results = {"baseline_mean": mae_base}

    for name, model in models.items():
        y_pred = model.predict(X_ext)
        mae_xy = mean_absolute_error(y_ext, y_pred, multioutput="raw_values")
        external_results[name] = mae_xy

    print("\n=========== EXTERNAL TEST RESULTS (normalized MAE) ===========")
    print("Model              | MAE_x      | MAE_y")
    print("-------------------+------------+-----------")
    for name, mae_xy in external_results.items():
        print(f"{name:18s} | {mae_xy[0]:10.4f} | {mae_xy[1]:10.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    # --- load and prepare train set ---
    df_train_long = parse_labels_dir(TRAIN_LABELS_DIR)
    df_train_wide = make_wide(df_train_long, dataset_name="TRAIN (train_on_all)")

    # --- load and prepare external test set ---
    df_test_long = parse_labels_dir(TEST_LABELS_DIR)
    df_test_wide = make_wide(df_test_long, dataset_name="TEST (prawn_2025_circ_small_v1)")

    # --- train models on train set ---
    models, feature_cols, target_cols = train_on_trainset(df_train_wide)

    # --- evaluate on external dataset ---
    evaluate_on_external(df_test_wide, models, feature_cols, target_cols)


if __name__ == "__main__":
    main()
