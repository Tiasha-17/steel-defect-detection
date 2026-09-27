"""Stratified 70/15/15 train/val/test split of the pooled NEU-CLS images.

No time dimension exists in this dataset (unlike the Olist delivery-risk
project, which had to split by order date to avoid leakage), so a stratified
random split is the correct choice here. Writes a manifest CSV rather than
copying image files, so the raw data stays as the single source of truth.

Run:
    python src/prepare_data.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

from model import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is whatever remains (0.15)


def collect_images() -> tuple[list[str], list[str]]:
    filepaths: list[str] = []
    labels: list[str] = []
    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {class_dir} -- run the data download steps in "
                "data/raw/README.md first."
            )
        for image_path in sorted(class_dir.glob("*.jpg")):
            filepaths.append(str(image_path.relative_to(ROOT)))
            labels.append(class_name)
    return filepaths, labels


def main() -> None:
    filepaths, labels = collect_images()
    print(f"Found {len(filepaths)} images across {len(CLASS_NAMES)} classes.")

    train_files, rest_files, train_labels, rest_labels = train_test_split(
        filepaths,
        labels,
        train_size=TRAIN_FRAC,
        random_state=SEED,
        stratify=labels,
    )
    val_size = VAL_FRAC / (1 - TRAIN_FRAC)
    val_files, test_files, val_labels, test_labels = train_test_split(
        rest_files,
        rest_labels,
        train_size=val_size,
        random_state=SEED,
        stratify=rest_labels,
    )

    rows = (
        [(f, l, "train") for f, l in zip(train_files, train_labels)]
        + [(f, l, "val") for f, l in zip(val_files, val_labels)]
        + [(f, l, "test") for f, l in zip(test_files, test_labels)]
    )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        writer.writerows(rows)

    for split_name, n in [
        ("train", len(train_files)),
        ("val", len(val_files)),
        ("test", len(test_files)),
    ]:
        print(f"{split_name}: {n} images")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
