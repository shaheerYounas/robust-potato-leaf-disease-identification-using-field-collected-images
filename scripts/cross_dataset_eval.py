"""
Cross-dataset generalization evaluation.

Tests the trained Hybrid CNN-Transformer on an external dataset
(e.g. PlantVillage) to measure how well the model generalises beyond
the Mendeley potato leaf dataset it was trained on.

Usage
-----
# Evaluate on PlantVillage potato subset:
python scripts/cross_dataset_eval.py \
    --data-dir data/external/PlantVillage/Potato \
    --checkpoint artifacts/phase_2_benchmarking/models/hybrid_cnn_transformer_best.pt

# Evaluate on any folder of class-labelled images (one subfolder per class):
python scripts/cross_dataset_eval.py \
    --data-dir /path/to/dataset \
    --class-map "Early_blight=Fungi,Late_blight=Phytopthora,healthy=Healthy"

The script produces:
  - Per-class precision / recall / F1
  - Overall accuracy and macro-F1
  - Confusion matrix saved as PNG
  - Results CSV for comparison
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from potato_leaf_inference import (
    HybridCNNTransformer,
    get_val_transforms,
    DEFAULT_CLASS_NAMES,
    DEFAULT_CHECKPOINT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_class_map(raw: str | None) -> dict[str, str] | None:
    """Parse 'FolderA=Bacteria,FolderB=Fungi,...' into a dict."""
    if not raw:
        return None
    mapping = {}
    for pair in raw.split(","):
        src, dst = pair.split("=")
        mapping[src.strip()] = dst.strip()
    return mapping


def load_external_dataset(
    data_dir: Path,
    class_map: dict[str, str] | None,
    model_classes: list[str],
) -> tuple[list[Path], list[int], list[str]]:
    """Walk *data_dir* expecting ImageFolder layout and return paths + labels.

    class_map translates external folder names → model class names.
    Folders that don't map to any model class are skipped with a warning.
    """
    image_paths: list[Path] = []
    labels: list[int] = []
    external_class_names: list[str] = []

    class_to_idx = {name: idx for idx, name in enumerate(model_classes)}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        folder_name = subdir.name
        mapped_name = folder_name
        if class_map:
            mapped_name = class_map.get(folder_name, folder_name)

        if mapped_name not in class_to_idx:
            print(f"[SKIP] Folder '{folder_name}' → '{mapped_name}' not in model classes, skipping.")
            continue

        label_idx = class_to_idx[mapped_name]
        class_images = [f for f in subdir.iterdir() if f.suffix.lower() in exts]
        print(f"  {folder_name} → {mapped_name} (idx={label_idx}): {len(class_images)} images")
        for p in class_images:
            image_paths.append(p)
            labels.append(label_idx)
        if mapped_name not in external_class_names:
            external_class_names.append(mapped_name)

    return image_paths, labels, external_class_names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    image_paths: list[Path],
    labels: list[int],
    class_names: list[str],
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """Run inference on all images and compute metrics."""
    transform = get_val_transforms()
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    total_time = 0.0

    model.eval()
    # Process in batches
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch_tensors = []
        for p in image_paths[start:end]:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        batch = torch.stack(batch_tensors).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        total_time += time.perf_counter() - t0

        preds = probs.argmax(axis=1).tolist()
        all_preds.extend(preds)
        all_probs.extend(probs)

    y_true = np.array(labels)
    y_pred = np.array(all_preds)

    # Only include classes present in the external dataset
    present_indices = sorted(set(labels))
    target_names = [class_names[i] for i in present_indices]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=present_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred,
        labels=present_indices,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)

    return {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": cm,
        "present_classes": target_names,
        "total_images": len(image_paths),
        "avg_latency_ms": (total_time / len(image_paths)) * 1000,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """Save confusion matrix as a PNG figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Cross-Dataset Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate model generalization on an external dataset.",
    )
    p.add_argument(
        "--data-dir", required=True, type=Path,
        help="Root of external dataset (ImageFolder layout: one subfolder per class).",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Model checkpoint (.pt).",
    )
    p.add_argument(
        "--class-map", type=str, default=None,
        help="Map external folder names to model classes, e.g. 'Early_blight=Fungi,Late_blight=Phytopthora'.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device override (cpu / cuda).",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cross_dataset",
        help="Directory for result files.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: --data-dir '{args.data_dir}' does not exist.")
        sys.exit(1)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    print(f"Loading model from {args.checkpoint} …")
    model = HybridCNNTransformer(n_classes=len(DEFAULT_CLASS_NAMES), freeze_backbone=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded on {device}")

    # Load dataset
    class_map = parse_class_map(args.class_map)
    print(f"\nScanning external dataset: {args.data_dir}")
    image_paths, labels, ext_classes = load_external_dataset(
        args.data_dir, class_map, DEFAULT_CLASS_NAMES,
    )
    if not image_paths:
        print("ERROR: No images found. Check --data-dir and --class-map.")
        sys.exit(1)
    print(f"  Total: {len(image_paths)} images across {len(ext_classes)} classes\n")

    # Evaluate
    print("Running inference …")
    results = evaluate(model, image_paths, labels, DEFAULT_CLASS_NAMES, device, args.batch_size)

    # Print report
    print("\n" + "=" * 60)
    print("CROSS-DATASET GENERALIZATION RESULTS")
    print("=" * 60)
    print(f"External dataset : {args.data_dir}")
    print(f"Total images     : {results['total_images']}")
    print(f"Accuracy         : {results['accuracy']:.4f}")
    print(f"Macro-F1         : {results['macro_f1']:.4f}")
    print(f"Avg latency      : {results['avg_latency_ms']:.2f} ms/image")
    print(f"\n{results['classification_report_text']}")

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix PNG
    save_confusion_matrix(
        results["confusion_matrix"],
        results["present_classes"],
        args.output_dir / "cross_dataset_confusion_matrix.png",
    )

    # Results CSV
    csv_path = args.output_dir / "cross_dataset_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["external_dataset", str(args.data_dir)])
        writer.writerow(["accuracy", f"{results['accuracy']:.4f}"])
        writer.writerow(["macro_f1", f"{results['macro_f1']:.4f}"])
        writer.writerow(["total_images", results["total_images"]])
        writer.writerow(["avg_latency_ms", f"{results['avg_latency_ms']:.2f}"])
        # Per-class F1
        for cls_name in results["present_classes"]:
            f1 = results["classification_report"].get(cls_name, {}).get("f1-score", 0)
            writer.writerow([f"f1_{cls_name}", f"{f1:.4f}"])
    print(f"Results CSV saved → {csv_path}")

    # Classification report text
    report_path = args.output_dir / "cross_dataset_classification_report.txt"
    report_path.write_text(results["classification_report_text"], encoding="utf-8")
    print(f"Classification report saved → {report_path}")


if __name__ == "__main__":
    main()
