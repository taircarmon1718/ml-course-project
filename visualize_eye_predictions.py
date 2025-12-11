"""
Visualize eye predictions for several models on ONE prawn example
from the external dataset (prawn_2025_circ_small_v1).

Each subplot shows:
- true tail / carapace / rostrum
- true eyes
- predicted eyes (for that model)
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

TRAIN_LABELS_DIR = os.path.join(TRAIN_ROOT, "labels")
TEST_LABELS_DIR  = os.path.join(TEST_ROOT, "labels")

TEST_IMAGES_DIR  = os.path.join(TEST_ROOT, "images")

# אפשר לכוון לדוגמה ספציפית:
EXAMPLE_IMAGE_STEM = None     # למשל: "GX010065_100_1575-jpg_gamma_jpg.rf.25d5..."
EXAMPLE_OBJECT_ID  = None     # למשל: 0
# אם תשאירי None, הוא יבחר דוגמה רנדומלית מהסט החיצוני.

# ============================================================
# HELPERS
# ============================================================
def parse_labels_dir(labels_dir: str) -> pd.DataFrame:
    """
    Parse YOLO keypoint .txt files from labels_dir **recursively**
    into long-format DataFrame.
    """
    rows = []

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
                    continue  # no keypoints

                cls = int(float(parts[0]))
                nums = list(map(float, parts[1:]))

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
    Convert long-format keypoints DF to wide-format:
       one row per (image_stem, object_id)
    Columns: carapace_x, carapace_y, eyes_x, eyes_y, rostrum_x, rostrum_y, tail_x, tail_y
    Indices mapping:
       0 = carapace-start
       1 = eyes
       2 = rostrum
       3 = tail
    """
    if df_long.empty:
        raise ValueError(f"{dataset_name}: no rows in df_long")

    df_visible = df_long[df_long["visibility"] == 2].copy()

    df_wide = df_visible.pivot_table(
        index=["image_stem", "object_id"],
        columns="keypoint_index",
        values=["x_norm", "y_norm"]
    )

    df_wide.columns = [f"{coord}{kpt_idx}" for (coord, kpt_idx) in df_wide.columns]
    df_wide = df_wide.reset_index()

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


def train_models_on_trainset(df_train_wide: pd.DataFrame):
    """
    Train models on train_on_all with internal split,
    and return trained models + mean_eyes (for baseline) + feature/target names.
    """
    feature_cols = [
        "carapace_x", "carapace_y",
        "rostrum_x", "rostrum_y",
        "tail_x", "tail_y",
    ]
    target_cols = ["eyes_x", "eyes_y"]

    X = df_train_wide[feature_cols].values
    y = df_train_wide[target_cols].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Internal split -> Train size:", len(X_tr), " | Test size:", len(X_te))

    # Baseline (mean eyes on train split)
    mean_eyes_train = y_tr.mean(axis=0)
    y_pred_base = np.tile(mean_eyes_train, (len(y_te), 1))
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

    return models, mean_eyes_train, feature_cols, target_cols


def find_image_path(images_root: str, image_stem: str) -> str:
    """
    Search recursively under images_root for a file whose name starts with image_stem.
    (Roboflow בד"כ שם התמונה = image_stem + '.jpg')
    """
    pattern = os.path.join(images_root, "**", image_stem + ".*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No image file found for stem '{image_stem}' under {images_root}")
    return candidates[0]


def choose_example(df_test_wide: pd.DataFrame):
    """
    Choose one example prawn from the external test set.
    If EXAMPLE_IMAGE_STEM / EXAMPLE_OBJECT_ID defined, use them, else random.
    """
    global EXAMPLE_IMAGE_STEM, EXAMPLE_OBJECT_ID

    if EXAMPLE_IMAGE_STEM is not None and EXAMPLE_OBJECT_ID is not None:
        mask = (
            (df_test_wide["image_stem"] == EXAMPLE_IMAGE_STEM) &
            (df_test_wide["object_id"] == EXAMPLE_OBJECT_ID)
        )
        subset = df_test_wide[mask]
        if subset.empty:
            raise ValueError("Requested example (image_stem, object_id) not found in df_test_wide.")
        row = subset.iloc[0]
    else:
        # pick a random example
        row = df_test_wide.sample(1, random_state=0).iloc[0]
        EXAMPLE_IMAGE_STEM = row["image_stem"]
        EXAMPLE_OBJECT_ID  = int(row["object_id"])

    print(f"\nChosen example: image_stem={EXAMPLE_IMAGE_STEM}, object_id={EXAMPLE_OBJECT_ID}")
    return row


def visualize_example(row, models, mean_eyes_train, feature_cols):
    """
    Create a single figure with:
      - First subplot: ONLY true eyes location
      - Other subplots: ONLY predicted eyes location for each model
    (all on the same image and same prawn)
    """
    # 1. find and load image from external dataset
    image_stem = row["image_stem"]
    img_path = find_image_path(TEST_IMAGES_DIR, image_stem)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"Loaded image for visualization: {img_path} (w={w}, h={h})")

    # 2. true eyes (normalized -> pixels)
    eyes_x_true = row["eyes_x"] * w
    eyes_y_true = row["eyes_y"] * h

    # 3. feature vector for this prawn
    X_example = row[feature_cols].values.reshape(1, -1)

    # 4. get predictions from all models (normalized coords)
    preds = {}

    # baseline: mean eyes from train split
    preds["baseline_mean"] = mean_eyes_train.copy()

    for name, model in models.items():
        y_pred = model.predict(X_example)[0]  # [eyes_x_norm, eyes_y_norm]
        preds[name] = y_pred

    # 5. define order and titles
    model_order = ["baseline_mean", "linear_regression", "random_forest", "mlp_small", "knn_k5"]
    titles = {
        "baseline_mean": "Baseline (mean eyes)",
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "mlp_small": "MLP",
        "knn_k5": "KNN (k=5)",
    }

    # keep only models that actually exist
    model_order = [m for m in model_order if m in preds]

    # first subplot = ground truth, then one per model
    n_models = len(model_order)
    n_subplots = n_models + 1  # 1 GT + N models

    fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4))

    if n_subplots == 1:
        axes = [axes]

    # --------- subplot 0: ONLY true eyes ---------
    ax0 = axes[0]
    ax0.imshow(img_rgb)
    ax0.axis("off")
    ax0.set_title("Ground Truth", fontsize=10)

    ax0.scatter([eyes_x_true], [eyes_y_true],
                c="magenta", s=60, marker="o", label="eyes (true)")

    ax0.legend(loc="lower right", fontsize=8)

    # --------- other subplots: ONLY predictions ---------
    for ax, model_name in zip(axes[1:], model_order):
        ax.imshow(img_rgb)
        ax.axis("off")
        ax.set_title(titles.get(model_name, model_name), fontsize=10)

        eyes_pred_norm = preds[model_name]
        eyes_x_pred = eyes_pred_norm[0] * w
        eyes_y_pred = eyes_pred_norm[1] * h

        ax.scatter([eyes_x_pred], [eyes_y_pred],
                   c="red", s=60, marker="x", label="eyes (pred)")

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "eye_prediction_models_comparison.png")
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved visualization figure to: {out_path}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    # 1. load and prepare train set
    df_train_long = parse_labels_dir(TRAIN_LABELS_DIR)
    df_train_wide = make_wide(df_train_long, dataset_name="TRAIN (train_on_all)")

    # 2. load and prepare external test set
    df_test_long = parse_labels_dir(TEST_LABELS_DIR)
    if df_test_long.empty:
        raise ValueError("External test set has no labels (df_test_long is empty).")
    df_test_wide = make_wide(df_test_long, dataset_name="TEST (prawn_2025_circ_small_v1)")

    # 3. train models on train set
    models, mean_eyes_train, feature_cols, target_cols = train_models_on_trainset(df_train_wide)

    # 4. choose example prawn from external test
    row_example = choose_example(df_test_wide)

    # 5. visualize
    visualize_example(row_example, models, mean_eyes_train, feature_cols)


if __name__ == "__main__":
    main()
