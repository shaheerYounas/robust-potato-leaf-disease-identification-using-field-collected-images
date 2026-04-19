from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
from torchvision import transforms

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

import torch.nn.functional as F

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGED_ROOT = DEFAULT_PROJECT_ROOT / "AdvancePractice"
DEFAULT_CHECKPOINT = DEFAULT_PROJECT_ROOT / "artifacts" / "phase_2_benchmarking" / "models" / "hybrid_cnn_transformer_best.pt"
DEFAULT_CLASS_INFO = PACKAGED_ROOT / "deployment" / "class_info.json"
DEFAULT_CENTROIDS = DEFAULT_PROJECT_ROOT / "artifacts" / "phase_2_benchmarking" / "models" / "leaf_centroids.pt"
FALLBACK_CHECKPOINT_PATHS = [
    DEFAULT_CHECKPOINT,
    PACKAGED_ROOT / "models" / "hybrid_cnn_transformer_best.pt",
]
FALLBACK_CLASS_INFO_PATHS = [
    DEFAULT_CLASS_INFO,
    DEFAULT_PROJECT_ROOT / "AdvancePractice" / "deployment" / "class_info.json",
    DEFAULT_PROJECT_ROOT / "Potato_Leaf_Disease_Storage" / "phase2" / "deployment" / "class_info.json",
]
FALLBACK_CENTROID_PATHS = [
    DEFAULT_CENTROIDS,
    PACKAGED_ROOT / "deployment" / "leaf_centroids.pt",
    DEFAULT_PROJECT_ROOT / "Potato_Leaf_Disease_Storage" / "phase2" / "models" / "leaf_centroids.pt",
    DEFAULT_PROJECT_ROOT / "Potato_Leaf_Disease_Storage" / "phase2" / "deployment" / "leaf_centroids.pt",
]

DEFAULT_CLASS_NAMES = [
    "Bacteria",
    "Fungi",
    "Healthy",
    "Nematode",
    "Pest",
    "Phytopthora",
    "Virus",
]

DEFAULT_DISEASE_INFO = {
    "Bacteria": "Bacterial leaf symptoms can appear as water-soaked or dark necrotic spots.",
    "Fungi": "Fungal infection often presents as expanding lesions with irregular boundaries.",
    "Healthy": "Leaf surface appears visually normal with no major disease symptoms.",
    "Nematode": "Nematode stress may appear through chlorosis, deformation, or weak tissue areas.",
    "Pest": "Pest damage can include chewing marks, punctures, or local tissue destruction.",
    "Phytopthora": "Phytophthora symptoms often include dark blight lesions and rapid tissue collapse.",
    "Virus": "Virus infection may produce mottling, mosaic patterns, curling, or stunted growth.",
}

# Plant-related ImageNet-1k class indices (for user-facing object labelling)
PLANT_IMAGENET_INDICES = {
    70, 301, 304, 311, 313, 315, 317, 324,
    738,
    936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946,
    947, 991, 992, 993, 994, 995, 996, 997,
    948, 949, 950, 951, 952, 953, 954, 955, 956, 957,
    958, 984, 985, 986, 987, 988, 989, 990, 998,
}

DEFAULT_LEAF_GATE_THRESHOLD = 0.45
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
MIN_GREEN_RATIO = 0.10


class EfficientNetTransfer(nn.Module):
    """EfficientNetB0 transfer learning model via timm."""
    def __init__(self, n_classes: int,
                 fine_tune: bool = False,
                 unfreeze_n_blocks: int = 2,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained_backbone,
            num_classes=0, global_pool=""
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        if fine_tune:
            blocks = list(self.backbone.blocks)
            for block_group in blocks[-unfreeze_n_blocks:]:
                for p in block_group.parameters():
                    p.requires_grad = True
            for p in self.backbone.conv_head.parameters():
                p.requires_grad = True
            for p in self.backbone.bn2.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_emb


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn, _ = self.attn(normed, normed, normed)
        x = x + self.drop1(attn)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 1280,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_trans_layers: int = 2,
        dropout_trans: float = 0.2,
        dropout_head: float = 0.3,
        freeze_backbone: bool = False,
        unfreeze_n_blocks: int = 2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool="",
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        if not freeze_backbone:
            for block_group in list(self.backbone.blocks)[-unfreeze_n_blocks:]:
                for param in block_group.parameters():
                    param.requires_grad = True
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
        self.pos_enc = LearnablePositionalEncoding(7 * 7, embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_trans)
                for _ in range(num_trans_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        patches = feats.flatten(2).transpose(1, 2)
        patches = self.pos_enc(patches)
        patches = self.transformer(patches)
        cls_vec = patches.mean(dim=1)
        return self.head(cls_vec)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    return get_val_transforms()(img.convert("RGB")).unsqueeze(0)


# ── Input brightness validation ──────────────────────────────────────────────
# Low-light is the primary robustness weakness (1.82% F1 drop in robustness analysis).
LOW_BRIGHTNESS_THRESH = 60.0
HIGH_BRIGHTNESS_THRESH = 220.0


def check_image_brightness(
    img: Image.Image,
    low_thresh: float = LOW_BRIGHTNESS_THRESH,
    high_thresh: float = HIGH_BRIGHTNESS_THRESH,
) -> dict:
    """Check mean brightness and flag if outside safe operational range."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    mean_brightness = float(gray.mean())
    if mean_brightness < low_thresh:
        level, msg = "WARNING_LOW_LIGHT", (
            f"Mean brightness {mean_brightness:.1f} < {low_thresh}. "
            "Low-light is the top robustness weakness — predictions may be unreliable."
        )
    elif mean_brightness > high_thresh:
        level, msg = "WARNING_OVEREXPOSED", (
            f"Mean brightness {mean_brightness:.1f} > {high_thresh}. "
            "Overexposed image — check source quality."
        )
    else:
        level, msg = "OK", f"Brightness {mean_brightness:.1f} is within safe range."
    return {"mean_brightness": round(mean_brightness, 2), "level": level, "message": msg}


def compute_green_ratio(img: Image.Image) -> float:
    """Estimate the fraction of vegetation-like green pixels in an image."""
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    green_mask = (
        (hue >= 60.0)
        & (hue <= 155.0)
        & (sat >= 0.20)
        & (val >= 0.15)
    )
    return round(float(green_mask.mean()), 4)


def compute_green_ratio_from_tensor(
    input_tensor: torch.Tensor,
    resize_to: tuple[int, int] | None = None,
) -> float:
    """Estimate vegetation-like green ratio from a normalized image tensor."""
    tensor = input_tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    rgb = tensor.permute(1, 2, 0).numpy()
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip((rgb * std) + mean, 0.0, 1.0)
    img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.BILINEAR)
    return compute_green_ratio(img)


def load_class_info(class_info_path: str | Path | None = None) -> dict:
    candidates = ([Path(class_info_path)] if class_info_path else []) + FALLBACK_CLASS_INFO_PATHS
    loaded_info = None
    loaded_path = None
    for path in candidates:
        if path.exists():
            loaded_info = json.loads(path.read_text(encoding="utf-8"))
            loaded_path = path
            break
    if loaded_info is not None:
        if "gate_config" not in loaded_info:
            for fallback in FALLBACK_CLASS_INFO_PATHS:
                if fallback == loaded_path or not fallback.exists():
                    continue
                fallback_info = json.loads(fallback.read_text(encoding="utf-8"))
                gate_config = fallback_info.get("gate_config")
                if gate_config:
                    loaded_info["gate_config"] = gate_config
                    break
        return loaded_info
    return {
        "final_model": "Hybrid CNN-Transformer",
        "class_names": DEFAULT_CLASS_NAMES,
        "class_to_idx": {name: idx for idx, name in enumerate(DEFAULT_CLASS_NAMES)},
        "disease_info": DEFAULT_DISEASE_INFO,
        "checkpoint_path": str(DEFAULT_CHECKPOINT),
    }


def load_centroids(
    centroids_path: str | Path | None = None,
    device: torch.device | None = None,
) -> dict | None:
    """Load pre-computed per-class feature centroids for the leaf gate."""
    candidates = ([Path(centroids_path)] if centroids_path else []) + FALLBACK_CENTROID_PATHS
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location=device or "cpu", weights_only=False)
    return None


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    class_info: dict,
) -> Path:
    candidates: list[Path] = []
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))

    info_path = class_info.get("checkpoint_path")
    if info_path:
        info_candidate = Path(info_path)
        candidates.append(info_candidate)
        if not info_candidate.is_absolute():
            candidates.append(DEFAULT_PROJECT_ROOT / info_candidate)
            candidates.append(PACKAGED_ROOT / info_candidate)

    candidates.extend(FALLBACK_CHECKPOINT_PATHS)

    for path in candidates:
        if path.exists():
            return path

    return Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT


def load_gate_model(device: torch.device | None = None) -> tuple[nn.Module | None, list[str] | None]:
    """Load EfficientNet-B0 with ImageNet-1k head for object labelling (Gate 2)."""
    try:
        gate_model = timm.create_model("efficientnet_b0", pretrained=True)
    except Exception:
        return None, None

    gate_model.eval()
    if device:
        gate_model = gate_model.to(device)
    # Get ImageNet class labels
    try:
        from torchvision.models import EfficientNet_B0_Weights
        categories = EfficientNet_B0_Weights.DEFAULT.meta["categories"]
    except Exception:
        categories = [f"class_{i}" for i in range(1000)]
    return gate_model, categories


def check_leaf_gate(
    input_tensor: torch.Tensor,
    model: nn.Module,
    centroids_data: dict,
    device: torch.device,
) -> dict:
    """Gate 1: Check if image looks like a potato leaf using backbone feature centroids.

    Returns dict with is_leaf, max_similarity, closest_class, threshold.
    """
    centroid_dict = centroids_data["centroids"]
    threshold = centroids_data.get("threshold", DEFAULT_LEAF_GATE_THRESHOLD)

    with torch.no_grad():
        feats = model.backbone(input_tensor.to(device))  # [1, 1280, 7, 7]
        pooled = feats.mean(dim=[2, 3])[0]                # [1280]

    similarities = {}
    for cls_name, centroid in centroid_dict.items():
        sim = F.cosine_similarity(
            pooled.unsqueeze(0),
            centroid.to(device).unsqueeze(0),
        ).item()
        similarities[cls_name] = sim

    closest_class = max(similarities, key=similarities.get)
    max_sim = similarities[closest_class]
    green_ratio = compute_green_ratio_from_tensor(input_tensor)
    smoothed_green_ratio = compute_green_ratio_from_tensor(input_tensor, resize_to=(32, 32))
    centroid_is_leaf = max_sim >= threshold
    hard_non_vegetation = smoothed_green_ratio < MIN_GREEN_RATIO
    soft_mismatch = (not centroid_is_leaf) and (not hard_non_vegetation)

    return {
        "is_leaf": not hard_non_vegetation,
        "centroid_is_leaf": centroid_is_leaf,
        "hard_non_vegetation": hard_non_vegetation,
        "soft_mismatch": soft_mismatch,
        "max_similarity": round(max_sim, 4),
        "closest_class": closest_class,
        "threshold": threshold,
        "green_ratio": green_ratio,
        "smoothed_green_ratio": smoothed_green_ratio,
        "all_similarities": {k: round(v, 4) for k, v in similarities.items()},
    }


def get_imagenet_label(
    input_tensor: torch.Tensor,
    gate_model: nn.Module,
    categories: list[str],
    device: torch.device,
    top_k: int = 3,
) -> list[dict]:
    """Gate 2: Get ImageNet object label for user-facing feedback."""
    with torch.no_grad():
        logits = gate_model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        top_vals, top_idx = probs.topk(top_k)

    results = []
    for val, idx in zip(top_vals, top_idx):
        idx_int = idx.item()
        results.append({
            "label": categories[idx_int] if idx_int < len(categories) else f"class_{idx_int}",
            "confidence": round(val.item(), 4),
            "is_plant_related": idx_int in PLANT_IMAGENET_INDICES,
        })
    return results


def load_model(
    checkpoint_path: str | Path | None = None,
    class_info_path: str | Path | None = None,
    device: str | None = None,
    centroids_path: str | Path | None = None,
) -> tuple[nn.Module, torch.device, dict]:
    info = load_class_info(class_info_path)
    class_names = info.get("class_names", DEFAULT_CLASS_NAMES)
    ckpt_path = resolve_checkpoint_path(checkpoint_path, info)
    active_device = get_device(device)
    model_name = info.get("final_model", "EfficientNetB0 (fine-tune)")
    if "hybrid" in model_name.lower():
        model = HybridCNNTransformer(
            n_classes=len(class_names),
            freeze_backbone=False,
            pretrained_backbone=False,
        ).to(active_device)
    else:
        model = EfficientNetTransfer(
            n_classes=len(class_names),
            fine_tune=True,
            pretrained_backbone=False,
        ).to(active_device)
    state_dict = torch.load(ckpt_path, map_location=active_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load centroids for leaf gate
    gate_cfg = info.get("gate_config", {})
    if centroids_path:
        cdata = load_centroids(centroids_path, active_device)
    elif gate_cfg.get("centroids_file"):
        cpath = Path(ckpt_path).parent / gate_cfg["centroids_file"]
        cdata = load_centroids(cpath, active_device)
    else:
        cdata = load_centroids(device=active_device)
    if cdata:
        info["_centroids"] = cdata

    return model, active_device, info


def predict_pil_image(
    img: Image.Image,
    model: nn.Module,
    device: torch.device,
    class_names: list[str],
    disease_info: dict[str, str],
    top_k: int = 3,
    validate_brightness: bool = True,
    centroids_data: dict | None = None,
    gate_model: nn.Module | None = None,
    gate_categories: list[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Hierarchical potato leaf disease prediction pipeline.

    Gate 1: Leaf similarity — cosine distance to class centroids (is it a leaf?)
    Gate 2: ImageNet labeller — human-readable object name (what IS it?)
    Gate 3: Confidence check — max softmax probability (is the model sure?)
    Classification: 7-class disease prediction
    """
    rgb_img = img.convert("RGB")
    brightness_check = check_image_brightness(rgb_img) if validate_brightness else None
    green_ratio = compute_green_ratio(rgb_img)
    input_tensor = preprocess_pil(rgb_img).to(device)

    # ── Gate 1: Leaf Similarity ───────────────────────────────────────────
    imagenet_labels = None
    detected_object = "unknown object"
    detector_says_plant = False
    if gate_model is not None and gate_categories is not None:
        imagenet_labels = get_imagenet_label(
            input_tensor, gate_model, gate_categories, device,
        )
        if imagenet_labels:
            detected_object = imagenet_labels[0]["label"]
            detector_says_plant = bool(imagenet_labels[0].get("is_plant_related"))

    leaf_gate = None
    leaf_gate_soft_failure = False
    leaf_gate_soft_reason = None
    if centroids_data is not None:
        leaf_gate = check_leaf_gate(input_tensor, model, centroids_data, device)
        obvious_non_vegetation = (not detector_says_plant) and leaf_gate.get("hard_non_vegetation", False)

        if obvious_non_vegetation:
            result = {
                "predicted_class": "Not a Potato Leaf",
                "confidence": 0.0,
                "rejected": True,
                "rejection_stage": "leaf_gate",
                "rejection_reason": (
                    f"Image does not resemble vegetation "
                    f"(green ratio {green_ratio:.3f} < {MIN_GREEN_RATIO:.3f}) "
                    f"and detector labelled it as {detected_object}."
                ),
                "detected_object": detected_object,
                "leaf_gate": leaf_gate,
                "top_k": [],
                "green_ratio": green_ratio,
            }
            if imagenet_labels:
                result["imagenet_labels"] = imagenet_labels
            if brightness_check:
                result["brightness_check"] = brightness_check
            return result

        if leaf_gate.get("soft_mismatch") or not leaf_gate["is_leaf"]:
            # Image is NOT a potato leaf — get ImageNet label for user feedback
            obvious_non_vegetation = (not detector_says_plant) and green_ratio < MIN_GREEN_RATIO
            if obvious_non_vegetation:
                result = {
                    "predicted_class": "Not a Potato Leaf",
                    "confidence": 0.0,
                    "rejected": True,
                    "rejection_stage": "leaf_gate",
                    "rejection_reason": (
                        f"Image does not resemble vegetation "
                        f"(green ratio {green_ratio:.3f} < {MIN_GREEN_RATIO:.3f}) "
                        f"and leaf similarity is too low "
                        f"({leaf_gate['max_similarity']:.3f} < "
                        f"{leaf_gate['threshold']:.3f})"
                    ),
                    "detected_object": detected_object,
                    "leaf_gate": leaf_gate,
                    "top_k": [],
                    "green_ratio": green_ratio,
                }
                if imagenet_labels:
                    result["imagenet_labels"] = imagenet_labels
                if brightness_check:
                    result["brightness_check"] = brightness_check
                return result

            leaf_gate_soft_failure = True
            leaf_gate_soft_reason = (
                f"Leaf similarity is low "
                f"({leaf_gate['max_similarity']:.3f} < {leaf_gate['threshold']:.3f}). "
                f"Detected object: {detected_object}. "
                f"This may be a leaf, but not a recognizable potato-leaf sample."
            )

    # ── Classification: 7-class disease model ─────────────────────────────
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for rank, idx in enumerate(top_indices, start=1):
        class_name = class_names[int(idx)]
        predictions.append({
            "rank": rank,
            "class_name": class_name,
            "probability": round(float(probs[idx]), 4),
            "disease_note": disease_info.get(class_name, ""),
        })

    predicted_class = predictions[0]["class_name"]
    confidence = predictions[0]["probability"]

    if leaf_gate_soft_failure:
        reason = leaf_gate_soft_reason
        if confidence < confidence_threshold:
            reason = (
                f"{reason} Model confidence is also low "
                f"({confidence:.3f} < {confidence_threshold:.3f})."
            )

        result = {
            "predicted_class": "Uncertain",
            "confidence": confidence,
            "rejected": True,
            "rejection_stage": "confidence_gate",
            "rejection_reason": reason,
            "best_guess": predicted_class,
            "detected_object": detected_object,
            "top_k": predictions,
            "green_ratio": green_ratio,
        }
        if imagenet_labels:
            result["imagenet_labels"] = imagenet_labels
        if leaf_gate:
            result["leaf_gate"] = leaf_gate
        if brightness_check:
            result["brightness_check"] = brightness_check
        return result

    # ── Gate 3: Confidence check ──────────────────────────────────────────
    if confidence < confidence_threshold:
        # Low confidence — might not be a potato leaf or ambiguous image
        result = {
            "predicted_class": "Uncertain",
            "confidence": confidence,
            "rejected": True,
            "rejection_stage": "confidence_gate",
            "rejection_reason": (
                f"Model confidence too low ({confidence:.3f} < "
                f"{confidence_threshold:.3f}) — possibly not a potato leaf"
            ),
            "best_guess": predicted_class,
            "detected_object": detected_object,
            "top_k": predictions,
            "green_ratio": green_ratio,
        }
        if imagenet_labels:
            result["imagenet_labels"] = imagenet_labels
        if leaf_gate:
            result["leaf_gate"] = leaf_gate
        if brightness_check:
            result["brightness_check"] = brightness_check
        return result

    # ── Accepted: return disease prediction ───────────────────────────────
    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "rejected": False,
        "top_k": predictions,
        "green_ratio": green_ratio,
    }
    if leaf_gate:
        result["leaf_gate"] = leaf_gate
    if brightness_check:
        result["brightness_check"] = brightness_check
    return result


def predict_image(
    image_path: str | Path,
    model: nn.Module,
    device: torch.device,
    class_names: list[str],
    disease_info: dict[str, str],
    top_k: int = 3,
    centroids_data: dict | None = None,
    gate_model: nn.Module | None = None,
    gate_categories: list[str] | None = None,
) -> dict:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        prediction = predict_pil_image(
            img, model, device, class_names, disease_info,
            top_k=top_k,
            centroids_data=centroids_data,
            gate_model=gate_model,
            gate_categories=gate_categories,
        )
    prediction["image_path"] = str(image_path)
    return prediction
