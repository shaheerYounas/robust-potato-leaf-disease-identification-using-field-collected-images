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

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = DEFAULT_PROJECT_ROOT / "artifacts" / "phase_2_benchmarking" / "models" / "efficientnet_b0_finetune_best.pt"
DEFAULT_CLASS_INFO = DEFAULT_PROJECT_ROOT / "submission_ready" / "final_package" / "deployment" / "class_info.json"

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


class EfficientNetTransfer(nn.Module):
    """EfficientNetB0 transfer learning model via timm."""
    def __init__(self, n_classes: int,
                 fine_tune: bool = False,
                 unfreeze_n_blocks: int = 2):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True,
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
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
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


def load_class_info(class_info_path: str | Path | None = None) -> dict:
    path = Path(class_info_path) if class_info_path else DEFAULT_CLASS_INFO
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "final_model": "EfficientNetB0 (fine-tune)",
        "class_names": DEFAULT_CLASS_NAMES,
        "class_to_idx": {name: idx for idx, name in enumerate(DEFAULT_CLASS_NAMES)},
        "disease_info": DEFAULT_DISEASE_INFO,
        "checkpoint_path": str(DEFAULT_CHECKPOINT),
    }


def load_model(
    checkpoint_path: str | Path | None = None,
    class_info_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[nn.Module, torch.device, dict]:
    info = load_class_info(class_info_path)
    class_names = info.get("class_names", DEFAULT_CLASS_NAMES)
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(info.get("checkpoint_path", DEFAULT_CHECKPOINT))
    active_device = get_device(device)
    model_name = info.get("final_model", "EfficientNetB0 (fine-tune)")
    if "hybrid" in model_name.lower():
        model = HybridCNNTransformer(n_classes=len(class_names), freeze_backbone=False).to(active_device)
    else:
        model = EfficientNetTransfer(n_classes=len(class_names), fine_tune=True).to(active_device)
    state_dict = torch.load(ckpt_path, map_location=active_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, active_device, info


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    return get_val_transforms()(img.convert("RGB")).unsqueeze(0)


def predict_pil_image(
    img: Image.Image,
    model: nn.Module,
    device: torch.device,
    class_names: list[str],
    disease_info: dict[str, str],
    top_k: int = 3,
) -> dict:
    input_tensor = preprocess_pil(img).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for rank, idx in enumerate(top_indices, start=1):
        class_name = class_names[int(idx)]
        predictions.append(
            {
                "rank": rank,
                "class_name": class_name,
                "probability": round(float(probs[idx]), 4),
                "disease_note": disease_info.get(class_name, ""),
            }
        )
    return {
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["probability"],
        "top_k": predictions,
    }


def predict_image(
    image_path: str | Path,
    model: nn.Module,
    device: torch.device,
    class_names: list[str],
    disease_info: dict[str, str],
    top_k: int = 3,
) -> dict:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        prediction = predict_pil_image(img, model, device, class_names, disease_info, top_k=top_k)
    prediction["image_path"] = str(image_path)
    return prediction
