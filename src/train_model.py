"""Train the steel defect classifier.

Approach: transfer learning via frozen-backbone feature extraction, not
fine-tuning end-to-end. 1,800 images is too small to fine-tune a CNN's
convolutional layers without overfitting, so the pretrained MobileNetV2
backbone is frozen and only a small classifier head is trained on its
cached embeddings -- this also makes CPU training take seconds, not hours.

Run:
    python src/train_model.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import CLASS_NAMES, DefectClassifier, build_transform

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"
MODEL_PATH = ROOT / "models" / "defect_classifier.pt"
METRICS_PATH = ROOT / "models" / "metrics.json"
REPORTS_DIR = ROOT / "reports"

SEED = 42
BATCH_SIZE = 32
HEAD_EPOCHS = 40
LEARNING_RATE = 1e-3
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(ROOT / row["filepath"]).convert("RGB")
        tensor = self.transform(image)
        label = LABEL_TO_IDX[row["label"]]
        return tensor, label


def load_manifest() -> dict[str, list[dict]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found -- run src/prepare_data.py first."
        )
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            splits[row["split"]].append(row)
    return splits


@torch.no_grad()
def extract_embeddings(model: DefectClassifier, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_embeddings, all_labels = [], []
    for images, labels in loader:
        all_embeddings.append(model.embed(images))
        all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_head(model: DefectClassifier, train_emb, train_labels, val_emb, val_labels) -> dict:
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_head_state = None
    history = []

    for epoch in range(1, HEAD_EPOCHS + 1):
        model.head.train()
        optimizer.zero_grad()
        logits = model.head(train_emb)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        model.head.eval()
        with torch.no_grad():
            train_acc = (logits.argmax(1) == train_labels).float().mean().item()
            val_logits = model.head(val_emb)
            val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()

        history.append({"epoch": epoch, "train_loss": loss.item(), "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.clone() for k, v in model.head.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

    model.head.load_state_dict(best_head_state)
    print(f"Best val accuracy: {best_val_acc:.3f} (head reverted to that checkpoint)")
    return {"history": history, "best_val_acc": best_val_acc}


@torch.no_grad()
def evaluate(model: DefectClassifier, embeddings, labels) -> dict:
    model.eval()
    logits = model.head(embeddings)
    preds = logits.argmax(1).tolist()
    truths = labels.tolist()

    report = classification_report(
        truths, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(truths, preds)
    return {"report": report, "confusion_matrix": cm}


def save_confusion_matrix(cm, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix -- Test Set")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    torch.manual_seed(SEED)
    splits = load_manifest()
    transform = build_transform()

    loaders = {
        name: DataLoader(
            ManifestImageDataset(rows, transform), batch_size=BATCH_SIZE, shuffle=False
        )
        for name, rows in splits.items()
    }

    print("Loading pretrained MobileNetV2 backbone (downloads weights on first run)...")
    model = DefectClassifier(pretrained=True)

    print("Extracting frozen-backbone embeddings for each split...")
    train_emb, train_labels = extract_embeddings(model, loaders["train"])
    val_emb, val_labels = extract_embeddings(model, loaders["val"])
    test_emb, test_labels = extract_embeddings(model, loaders["test"])

    print(f"Training classifier head for {HEAD_EPOCHS} epochs...")
    train_head(model, train_emb, train_labels, val_emb, val_labels)

    print("Evaluating on held-out test set...")
    results = evaluate(model, test_emb, test_labels)
    report = results["report"]
    cm = results["confusion_matrix"]

    print(f"Test accuracy: {report['accuracy']:.3f}")
    save_confusion_matrix(cm, REPORTS_DIR / "confusion_matrix.png")

    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(
            classification_report(
                test_labels.tolist(),
                model.head(test_emb).argmax(1).tolist(),
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved model to {MODEL_PATH} ({model_size_mb:.1f} MB)")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "backbone": "mobilenet_v2 (frozen, ImageNet-pretrained)",
        "class_names": CLASS_NAMES,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
        "model_size_mb": round(model_size_mb, 2),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
