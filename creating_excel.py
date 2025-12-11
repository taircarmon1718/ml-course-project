import os
import glob
import pandas as pd

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
PROJECT_ROOT = r"C:\Users\carmonta\Desktop\ML_Course_project"
LABELS_DIR = os.path.join(PROJECT_ROOT, "train_on_all", "labels")
OUTPUT_XLSX = os.path.join(PROJECT_ROOT, "keypoints_from_label_txt.xlsx")

def parse_labels(label_dir):
    rows = []

    # all .txt files in labels dir
    pattern = os.path.join(label_dir, "*.txt")
    label_files = glob.glob(pattern)
    print(f"Found {len(label_files)} label files.")

    for label_path in label_files:
        img_stem = os.path.splitext(os.path.basename(label_path))[0]  # "000001", etc.

        with open(label_path, "r") as f:
            for obj_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # YOLO keypoints format (Ultralytics):
                # class cx cy w h kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
                if len(parts) <= 5:
                    continue  # no keypoints

                cls = int(float(parts[0]))
                nums = list(map(float, parts[1:]))

                if len(nums) < 4:
                    continue

                # first 4 are bbox
                bbox = nums[:4]
                kpts = nums[4:]

                # keypoints should come in triplets (x, y, v)
                if len(kpts) % 3 != 0:
                    print(f"Warning: keypoints len not divisible by 3 in {label_path}")
                    continue

                n_kpts = len(kpts) // 3

                for kpt_idx in range(n_kpts):
                    x = kpts[3 * kpt_idx + 0]
                    y = kpts[3 * kpt_idx + 1]
                    v = kpts[3 * kpt_idx + 2]

                    rows.append({
                        "image_stem": img_stem,
                        "object_id": obj_id,      # which row in the label file
                        "class_id": cls,
                        "keypoint_index": kpt_idx,
                        "x_norm": x,              # normalized 0–1
                        "y_norm": y,
                        "visibility": v
                    })

    print(f"Collected {len(rows)} keypoints.")
    return pd.DataFrame(rows)


def main():
    df = parse_labels(LABELS_DIR)

    if df.empty:
        print("No keypoints found. Check that labels contain keypoint annotations.")
        return

    print("Saving Excel:", OUTPUT_XLSX)
    df.to_excel(OUTPUT_XLSX, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
