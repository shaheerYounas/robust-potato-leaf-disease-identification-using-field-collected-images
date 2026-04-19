"""
Retrain the Hybrid CNN-Transformer with an 8th "Unknown" class.

Strategy: Since the EfficientNet backbone is frozen, we pre-extract all
backbone features once (slow step), then train only the Transformer + head
on cached features (fast step). This makes CPU training feasible.

USAGE
-----
1. Ensure data/processed/augmented_dataset/Unknown/ has negative images.
2. Run: python scripts/retrain_with_unknown_class.py
3. Model saved to: artifacts/phase_2_benchmarking/models/hybrid_cnn_transformer_with_ood.pt
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "augmented_dataset"
CHECKPOINT_IN = PROJECT_ROOT / "artifacts" / "phase_2_benchmarking" / "models" / "hybrid_cnn_transformer_best.pt"
CHECKPOINT_OUT = PROJECT_ROOT / "artifacts" / "phase_2_benchmarking" / "models" / "hybrid_cnn_transformer_with_ood.pt"
CLASS_INFO_OUT = PROJECT_ROOT / "AdvancePractice" / "deployment" / "class_info_with_ood.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
EXTRACT_BATCH = 8     # batch size for feature extraction (one-time)
TRAIN_BATCH = 64      # batch size for cached-feature training (fast)
EPOCHS = 20
LR = 5e-5
LABEL_SMOOTHING = 0.1
PATIENCE = 5
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformerHead(nn.Module):
    """Standalone Transformer + classification head that operates on cached backbone features."""
    def __init__(self, embed_dim=1280, num_patches=49, n_classes=8,
                 num_heads=8, ff_dim=2048, num_layers=2,
                 dropout_trans=0.2, dropout_head=0.3):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(embed_dim, num_heads, ff_dim, dropout_trans))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(dropout_head),
            nn.Linear(256, n_classes),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = patches + self.pos_emb
        for block in self.blocks:
            x = block(x)
        cls_vec = x.mean(dim=1)
        return self.head(cls_vec)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block matching the original HybridCNNTransformer architecture."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn, _ = self.attn(normed, normed, normed)
        x = x + self.drop1(attn)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


def extract_backbone_features(backbone, dataset, device, batch_size=8):
    """Run all images through the frozen backbone once, return (features, labels)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats, all_labels = [], []
    n_batches = len(loader)
    t0 = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader, 1):
            feats = backbone(images.to(device))          # [B, 1280, 7, 7]
            patches = feats.flatten(2).transpose(1, 2)   # [B, 49, 1280]
            all_feats.append(patches.cpu())
            all_labels.append(labels)
            if i % 25 == 0 or i == n_batches:
                elapsed = time.time() - t0
                eta = elapsed / i * (n_batches - i)
                print(f"  Extracting: {i}/{n_batches} batches ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    return torch.cat(all_feats), torch.cat(all_labels)


def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    from potato_leaf_inference import HybridCNNTransformer

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Verify Unknown folder ─────────────────────────────────────────────
    unknown_dir = DATASET_DIR / "Unknown"
    if not unknown_dir.exists():
        print(f"\nERROR: '{unknown_dir}' does not exist.")
        print("Create this folder and add 200+ images of non-potato-leaf objects.\n")
        sys.exit(1)

    n_unknown = len(list(unknown_dir.glob("*")))
    print(f"Found {n_unknown} images in Unknown class")

    # ── Dataset split ─────────────────────────────────────────────────────
    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    full_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=val_tf)
    class_names = full_dataset.classes
    n_classes = len(class_names)

    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    print(f"Classes ({n_classes}): {class_names}")
    print(f"Total: {n} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ── Load pre-trained backbone ─────────────────────────────────────────
    print("\nLoading pre-trained model backbone...")
    old_model = HybridCNNTransformer(n_classes=7, freeze_backbone=False)
    state_dict = torch.load(CHECKPOINT_IN, map_location=device, weights_only=True)
    old_model.load_state_dict(state_dict)
    old_model.eval()

    backbone = old_model.backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    # ── Extract features (one-time) ───────────────────────────────────────
    print("\n=== Extracting backbone features (one-time) ===")
    t_start = time.time()
    all_feats, all_labels = extract_backbone_features(
        backbone, full_dataset, device, batch_size=EXTRACT_BATCH
    )
    print(f"Feature extraction done in {time.time() - t_start:.0f}s")
    print(f"Features shape: {all_feats.shape}")  # [N, 49, 1280]

    # Split into train/val/test
    train_feats, train_labels = all_feats[train_idx], all_labels[train_idx]
    val_feats, val_labels = all_feats[val_idx], all_labels[val_idx]
    test_feats, test_labels = all_feats[test_idx], all_labels[test_idx]

    # Free backbone memory
    del backbone, old_model, all_feats, all_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Data loaders on cached features ───────────────────────────────────
    train_targets_np = train_labels.numpy()
    class_counts = np.bincount(train_targets_np, minlength=n_classes)
    weights_per_class = 1.0 / (class_counts + 1e-6)
    sample_weights = weights_per_class[train_targets_np]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    test_ds = TensorDataset(test_feats, test_labels)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_BATCH, shuffle=False)

    # ── Build Transformer head ────────────────────────────────────────────
    trans_head = TransformerHead(
        embed_dim=1280, num_patches=49, n_classes=n_classes,
    ).to(device)

    # Copy pre-trained transformer + head weights (first 7 classes)
    old_state = state_dict  # already loaded
    new_state = trans_head.state_dict()

    # Map original model keys to our TransformerHead keys
    # Original: transformer.0.norm1.weight -> blocks.0.norm1.weight
    for key in list(new_state.keys()):
        if key == "pos_emb":
            if "pos_enc.pos_emb" in old_state:
                new_state[key] = old_state["pos_enc.pos_emb"]
        elif key.startswith("blocks."):
            # blocks.0.norm1.weight -> transformer.0.norm1.weight
            old_key = key.replace("blocks.", "transformer.", 1)
            if old_key in old_state and old_state[old_key].shape == new_state[key].shape:
                new_state[key] = old_state[old_key]
        elif key.startswith("head."):
            if key in old_state:
                old_val = old_state[key]
                new_val = new_state[key]
                if old_val.shape == new_val.shape:
                    new_state[key] = old_val
                elif len(old_val.shape) >= 1 and old_val.shape[0] == 7 and new_val.shape[0] == 8:
                    # Copy first 7 rows, keep 8th randomly initialized
                    new_state[key][:7] = old_val
                else:
                    new_state[key] = old_val

    trans_head.load_state_dict(new_state)
    print("Loaded pre-trained transformer+head weights; expanded for Unknown class")

    # Count trainable params
    n_params = sum(p.numel() for p in trans_head.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Training ──────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(trans_head.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    print(f"\n=== Training Transformer Head ({EPOCHS} epochs) ===")
    for epoch in range(1, EPOCHS + 1):
        trans_head.train()
        running_loss, correct, total = 0.0, 0, 0
        t_ep = time.time()

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = trans_head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trans_head.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        trans_head.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = trans_head(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * feats.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        scheduler.step(val_loss)

        ep_time = time.time() - t_ep
        print(f"Epoch {epoch:2d}/{EPOCHS} ({ep_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in trans_head.state_dict().items()}
            print(f"  -> New best (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Reassemble full model & save ──────────────────────────────────────
    print("\n=== Assembling full model ===")
    full_model = HybridCNNTransformer(n_classes=n_classes, freeze_backbone=True)

    # Load original backbone weights
    full_state = full_model.state_dict()
    for key, val in state_dict.items():
        if key.startswith("backbone.") and key in full_state:
            full_state[key] = val

    # Map TransformerHead weights back to HybridCNNTransformer keys
    full_state["pos_enc.pos_emb"] = best_state["pos_emb"]
    for key in best_state:
        if key.startswith("blocks."):
            full_key = key.replace("blocks.", "transformer.", 1)
            if full_key in full_state:
                full_state[full_key] = best_state[key]
        elif key.startswith("head."):
            if key in full_state:
                full_state[key] = best_state[key]

    full_model.load_state_dict(full_state)
    full_model.eval()

    # Save
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_model.state_dict(), CHECKPOINT_OUT)
    print(f"Saved full model to {CHECKPOINT_OUT}")

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n=== Test Evaluation ===")
    trans_head.load_state_dict(best_state)
    trans_head.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            preds = trans_head(feats).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    from sklearn.metrics import classification_report
    print(classification_report(all_true, all_preds, target_names=class_names, zero_division=0))

    # ── Save class_info ───────────────────────────────────────────────────
    class_info = {
        "final_model": "Hybrid CNN-Transformer (with OOD)",
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "disease_info": {
            "Bacteria": "Bacterial leaf symptoms can appear as water-soaked or dark necrotic spots.",
            "Fungi": "Fungal infection often presents as expanding lesions with irregular boundaries.",
            "Healthy": "Leaf surface appears visually normal with no major disease symptoms.",
            "Nematode": "Nematode stress may appear through chlorosis, deformation, or weak tissue areas.",
            "Pest": "Pest damage can include chewing marks, punctures, or local tissue destruction.",
            "Phytopthora": "Phytophthora symptoms often include dark blight lesions and rapid tissue collapse.",
            "Unknown": "Image does not appear to be a potato leaf.",
            "Virus": "Virus infection may produce mottling, mosaic patterns, curling, or stunted growth.",
        },
        "checkpoint_path": str(CHECKPOINT_OUT.relative_to(PROJECT_ROOT)),
    }
    CLASS_INFO_OUT.parent.mkdir(parents=True, exist_ok=True)
    CLASS_INFO_OUT.write_text(json.dumps(class_info, indent=2), encoding="utf-8")
    print(f"Saved class info to {CLASS_INFO_OUT}")
    print("\nDone! To use the new model, update your checkpoint and class_info paths.")


if __name__ == "__main__":
    main()
