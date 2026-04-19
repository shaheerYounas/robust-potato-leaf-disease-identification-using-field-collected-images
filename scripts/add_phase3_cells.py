"""
Replace Phase 3 cells in the notebook.
Removes old Unknown-class cells (indices 63+) and adds new hierarchical
leaf-gate cells that require NO retraining.

Run once:  python scripts/add_phase3_cells.py
"""
import json, textwrap

NB_PATH = "Notebook/Advance_Practice_Potato_Leaf.ipynb"


def _src(text: str) -> list[str]:
    """Convert a multi-line string to notebook source list."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Cell definitions
# ══════════════════════════════════════════════════════════════════════════════

CELLS = [
    # ── 0: Section header ─────────────────────────────────────────────────
    md("""\
## Phase 3 — Hierarchical Leaf Gate for Real-World Deployment

Instead of training an extra "Unknown" class (which needs representative negative
data and may not generalise), we build a **three-stage gating pipeline** that
requires **zero additional training**:

| Stage | What it checks | How |
|-------|---------------|-----|
| **Gate 1 – Leaf similarity** | Is this image similar to ANY known potato-leaf class? | Cosine similarity between backbone features and per-class centroids |
| **Gate 2 – ImageNet labeller** | What IS this object? (user-facing UX) | EfficientNet-B0 with original ImageNet-1k head |
| **Gate 3 – Confidence check** | Is the model confident enough? | Max softmax + Shannon entropy |

The existing **7-class** disease model stays unchanged.  We only need to compute
class centroids (one forward pass through the training set) and pick a threshold."""),

    # ── 1: Compute centroids ──────────────────────────────────────────────
    code("""\
# -- Phase 3a: Compute per-class feature centroids from trained backbone ----------
import torch.nn.functional as F

CENTROID_BATCH = 32

# ── Load best 7-class model ──────────────────────────────────────────────────────
_best_ckpt_path = os.path.join(MODEL_DIR, "hybrid_cnn_transformer_best.pt")
assert os.path.exists(_best_ckpt_path), f"Checkpoint not found: {_best_ckpt_path}"

_gate_model = HybridCNNTransformer(n_classes=len(cls_names), freeze_backbone=False).to(DEVICE)
_gate_model.load_state_dict(torch.load(_best_ckpt_path, map_location=DEVICE, weights_only=True))
_gate_model.eval()
_backbone = _gate_model.backbone

# ── Dataset (same as Phase 2 — 7 disease classes) ────────────────────────────────
_centroid_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
_centroid_ds = datasets.ImageFolder(str(DATASET_ROOT), transform=_centroid_tf)
_centroid_loader = DataLoader(_centroid_ds, batch_size=CENTROID_BATCH, shuffle=False, num_workers=0)

assert _centroid_ds.classes == cls_names, (
    f"Dataset classes {_centroid_ds.classes} != expected {cls_names}"
)

# ── Extract backbone features per class ─────────────────────────────────────────
print("=== Computing per-class feature centroids ===")
_class_feats = {c: [] for c in cls_names}
t0 = time.time()

with torch.no_grad():
    for i, (imgs, labels) in enumerate(_centroid_loader, 1):
        feats = _backbone(imgs.to(DEVICE))          # [B, 1280, 7, 7]
        pooled = feats.mean(dim=[2, 3])              # [B, 1280]
        for feat, lbl in zip(pooled.cpu(), labels):
            _class_feats[cls_names[lbl.item()]].append(feat)
        if i % 20 == 0 or i == len(_centroid_loader):
            print(f"  Batch {i}/{len(_centroid_loader)}")

# ── Compute centroids ────────────────────────────────────────────────────────────
centroids = {}
for cls_name, feat_list in _class_feats.items():
    stacked = torch.stack(feat_list)            # [N_cls, 1280]
    centroids[cls_name] = stacked.mean(dim=0)   # [1280]
    print(f"  {cls_name:>12}: {len(feat_list)} images -> centroid shape {centroids[cls_name].shape}")

elapsed = time.time() - t0
print(f"\\nCentroid computation done in {elapsed:.1f}s ({len(_centroid_ds)} images)")

del _class_feats, _backbone
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()"""),

    # ── 2: Calibration heading ────────────────────────────────────────────
    md("""\
### Phase 3b — Gate Threshold Calibration

We compute cosine similarity between test potato-leaf images and the centroids
(positive distribution), then do the same for random CIFAR-10 objects (negative
distribution).  A clean separation ⇒ a reliable threshold."""),

    # ── 3: Calibration code ───────────────────────────────────────────────
    code("""\
# -- Phase 3b: Calibrate leaf-gate threshold with positive & negative samples -----
from torchvision.datasets import CIFAR10

def _max_centroid_sim(feat_vec, centroid_dict):
    \"\"\"Cosine similarity between feat_vec [1280] and nearest centroid.\"\"\"
    sims = []
    for c in centroid_dict.values():
        sim = F.cosine_similarity(feat_vec.unsqueeze(0), c.unsqueeze(0)).item()
        sims.append(sim)
    return max(sims)

# ── Positive: all potato-leaf images ─────────────────────────────────────────────
print("Computing similarities for potato-leaf images...")
_pos_loader = DataLoader(_centroid_ds, batch_size=CENTROID_BATCH, shuffle=False, num_workers=0)
_pos_sims = []
with torch.no_grad():
    for imgs, _ in _pos_loader:
        feats = _gate_model.backbone(imgs.to(DEVICE)).mean(dim=[2, 3]).cpu()
        for f in feats:
            _pos_sims.append(_max_centroid_sim(f, centroids))

# ── Negative: CIFAR-10 objects ───────────────────────────────────────────────────
print("Downloading CIFAR-10 for negative calibration...")
_cifar_cache = PROJECT_ROOT / "data" / "cache"
_cifar_ds = CIFAR10(root=str(_cifar_cache), train=False, download=True, transform=_centroid_tf)
_neg_loader = DataLoader(_cifar_ds, batch_size=CENTROID_BATCH, shuffle=True, num_workers=0)

_neg_sims = []
with torch.no_grad():
    for imgs, _ in _neg_loader:
        feats = _gate_model.backbone(imgs.to(DEVICE)).mean(dim=[2, 3]).cpu()
        for f in feats:
            _neg_sims.append(_max_centroid_sim(f, centroids))
        if len(_neg_sims) >= 500:
            break

_pos_arr = np.array(_pos_sims)
_neg_arr = np.array(_neg_sims[:500])

# ── Find optimal threshold ───────────────────────────────────────────────────────
# Threshold = value that keeps >=99% of positives while rejecting most negatives
_pos_p1  = float(np.percentile(_pos_arr, 1))   # 1st percentile of positives
_neg_p99 = float(np.percentile(_neg_arr, 99))  # 99th percentile of negatives
_threshold = round((_pos_p1 + _neg_p99) / 2, 4)

# Compute accuracy at chosen threshold
tp = int((_pos_arr >= _threshold).sum())
fp = int((_neg_arr >= _threshold).sum())
tn = int((_neg_arr <  _threshold).sum())
fn = int((_pos_arr <  _threshold).sum())
precision = tp / (tp + fp + 1e-9)
recall    = tp / (tp + fn + 1e-9)

print(f"\\n=== Gate Calibration Results ===")
print(f"Positive (potato leaf) — min: {_pos_arr.min():.4f}  1st %%ile: {_pos_p1:.4f}  mean: {_pos_arr.mean():.4f}")
print(f"Negative (CIFAR-10)    — max: {_neg_arr.max():.4f}  99th %%ile: {_neg_p99:.4f}  mean: {_neg_arr.mean():.4f}")
print(f"Chosen threshold: {_threshold}")
print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
print(f"  True positives:  {tp}/{len(_pos_arr)} potato leaves pass")
print(f"  False positives: {fp}/{len(_neg_arr)} objects wrongly pass")
print(f"  False negatives: {fn}/{len(_pos_arr)} potato leaves blocked")

# ── Plot distributions ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(_pos_arr, bins=80, alpha=0.6, label=f"Potato Leaf (n={len(_pos_arr)})", color="green", density=True)
ax.hist(_neg_arr, bins=80, alpha=0.6, label=f"Non-Leaf Objects (n={len(_neg_arr)})", color="red", density=True)
ax.axvline(_threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {_threshold:.3f}")
ax.set_xlabel("Max Cosine Similarity to Nearest Centroid")
ax.set_ylabel("Density")
ax.set_title("Leaf Gate: Similarity Distribution — Potato Leaves vs Random Objects")
ax.legend()
plt.tight_layout()
_gate_plot = os.path.join(PLOT_DIR, "leaf_gate_calibration.png")
fig.savefig(_gate_plot, dpi=150)
plt.show()
print(f"[INFO] Gate calibration plot saved -> {_gate_plot}")

del _gate_model  # free memory
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()"""),

    # ── 4: Save heading ──────────────────────────────────────────────────
    md("""\
### Phase 3c — Save Gate Artifacts for Deployment"""),

    # ── 5: Save centroids + class_info ────────────────────────────────────
    code("""\
# -- Phase 3c: Save leaf-gate artifacts for deployment ----------------------------

# ── Save centroids + threshold ───────────────────────────────────────────────────
_centroids_payload = {
    "centroids": centroids,
    "threshold": _threshold,
    "class_names": cls_names,
}
_centroids_path = os.path.join(MODEL_DIR, "leaf_centroids.pt")
torch.save(_centroids_payload, _centroids_path)
print(f"[INFO] Centroids + threshold saved -> {_centroids_path}")
print(f"       Threshold: {_threshold}")
print(f"       Classes:   {cls_names}")

# ── ImageNet class labels for the object labeller (Gate 2 UX) ────────────────────
# We include the known plant-related ImageNet class indices so the deployment
# code can identify whether ImageNet's top prediction is plant-like.
from torchvision.models import EfficientNet_B0_Weights as _EBW
_imagenet_categories = _EBW.DEFAULT.meta["categories"]

# Curated set: vegetables, fruits, flowers, fungi, leaf-dwelling insects, plant pots
PLANT_IMAGENET_INDICES = sorted({
    70, 301, 304, 311, 313, 315, 317, 324,                  # leaf-associated insects
    738,                                                      # pot / planter
    936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946,  # vegetables
    947, 991, 992, 993, 994, 995, 996, 997,                  # mushrooms / fungi
    948, 949, 950, 951, 952, 953, 954, 955, 956, 957,        # fruits
    958, 984, 985, 986, 987, 988, 989, 990, 998,             # flowers / plants / hay
})

# ── Save class_info.json (7-class, with gate config) ─────────────────────────────
class_info = {
    "final_model": "Hybrid CNN-Transformer",
    "class_names": cls_names,
    "class_to_idx": {name: idx for idx, name in enumerate(cls_names)},
    "disease_info": {
        "Bacteria":    "Bacterial leaf symptoms — water-soaked or dark necrotic spots.",
        "Fungi":       "Fungal infection — expanding lesions with irregular boundaries.",
        "Healthy":     "Leaf appears visually normal with no major disease symptoms.",
        "Nematode":    "Nematode stress — chlorosis, deformation, or weak tissue areas.",
        "Pest":        "Pest damage — chewing marks, punctures, or local tissue destruction.",
        "Phytopthora": "Phytophthora — dark blight lesions and rapid tissue collapse.",
        "Virus":       "Virus infection — mottling, mosaic patterns, curling, or stunted growth.",
    },
    "gate_config": {
        "leaf_gate_threshold": _threshold,
        "confidence_threshold": 0.30,
        "entropy_threshold": 1.8,
        "centroids_file": "leaf_centroids.pt",
        "plant_imagenet_indices": PLANT_IMAGENET_INDICES,
    },
}

# Save to both MODEL_DIR and DEPLOY_DIR
for _dst in [
    os.path.join(MODEL_DIR, "class_info.json"),
    os.path.join(str(DEPLOY_DIR), "class_info.json"),
]:
    with open(_dst, "w", encoding="utf-8") as _f:
        json.dump(class_info, _f, indent=2)
    print(f"[INFO] class_info.json -> {_dst}")

# Copy centroids to deploy dir
import shutil as _shutil
_deploy_centroids = os.path.join(str(DEPLOY_DIR), "leaf_centroids.pt")
_shutil.copy2(_centroids_path, _deploy_centroids)
print(f"[INFO] Centroids copied -> {_deploy_centroids}")

print("\\n[DONE] Phase 3 complete — leaf gate ready for deployment (zero retraining).")"""),

    # ── 6: Drive structure ────────────────────────────────────────────────
    md("""\
### Google Drive Folder Structure

To run this notebook on **Google Colab**, upload the following to your Google Drive:

```
MyDrive/
└── data/
    └── raw/
        └── Potato Leaf Disease Dataset in Uncontrolled Environment/
            └── Potato Leaf Disease Dataset in Uncontrolled Environment/
                ├── Bacteria/      (569 images)
                ├── Fungi/         (748 images)
                ├── Healthy/       (201 images)
                ├── Nematode/      (68 images)
                ├── Pest/          (611 images)
                ├── Phytopthora/   (347 images)
                └── Virus/         (532 images)
```

**No "Unknown" images needed!** The leaf gate uses the trained model's own
feature space — no extra negative training data required.

The `Potato_Leaf_Disease_Storage/` folder is created automatically by the notebook.
Model checkpoints, centroids, and plots save to Drive so progress survives
Colab disconnections."""),
]


# ══════════════════════════════════════════════════════════════════════════════
# Apply to notebook
# ══════════════════════════════════════════════════════════════════════════════

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Remove old Phase 3 cells (indices 63 onward — the Unknown-class approach)
original_count = len(nb["cells"])
nb["cells"] = nb["cells"][:63]  # keep only the original Phase 1+2 cells

# Append new cells
nb["cells"].extend(CELLS)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Removed {original_count - 63} old Phase 3 cells, added {len(CELLS)} new cells.")
print(f"Total: {len(nb['cells'])} cells.")
