"""Export pre-trained MobileNetV3-Small (ImageNet-1k) as a leaf/object detector for Android.

This model serves as Gate 0 in the Android app: it identifies what the image
contains and rejects anything that is not a plant / leaf BEFORE the disease
classification model runs.
"""

import os
import json

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import onnx
from onnxconverter_common import float16

# ── Plant-related ImageNet-1k class indices ──────────────────────────────────
# Covers vegetables, fruits, flowers, plants, fungi, and greenery.
# Same set used in potato_leaf_inference.py Gate-2 object labelling.
PLANT_IMAGENET_INDICES = sorted([
    # Vegetables & leafy things
    738,  # head cabbage
    # Fruits / produce (many overlap with plant appearance)
    936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947,
    948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958,
    # Flowers & ornamental plants
    984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995,
    996, 997, 998,
    # Fungi / mushrooms (visually plant-like)
    70,
    # Misc plants / nature
    301, 304, 311, 313, 315, 317, 324,
])

ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "android_build",
    "PotatoLeafApp", "app", "src", "main", "assets",
)


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # ── 1. Load pre-trained MobileNetV3-Small ────────────────────────────
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    categories = list(weights.meta["categories"])
    print(f"Loaded MobileNetV3-Small  ({len(categories)} classes)")

    # ── 2. Export to ONNX (float32) ──────────────────────────────────────
    dummy = torch.randn(1, 3, 224, 224)
    onnx_fp32 = os.path.join(ASSETS_DIR, "leaf_detector.onnx")

    torch.onnx.export(
        model, dummy, onnx_fp32,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    fp32_mb = os.path.getsize(onnx_fp32) / 1024 / 1024
    print(f"FP32 ONNX: {fp32_mb:.1f} MB")

    # ── 3. Convert to FP16 ──────────────────────────────────────────────
    onnx_model = onnx.load(onnx_fp32)
    fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    fp16_path = os.path.join(ASSETS_DIR, "leaf_detector_fp16.onnx")
    onnx.save(fp16_model, fp16_path)
    fp16_mb = os.path.getsize(fp16_path) / 1024 / 1024
    print(f"FP16 ONNX: {fp16_mb:.1f} MB")

    # Remove fp32 intermediate
    os.remove(onnx_fp32)

    # ── 4. Save detector config ─────────────────────────────────────────
    detector_config = {
        "model": "MobileNetV3-Small",
        "input_size": 224,
        "class_names": categories,
        "plant_indices": PLANT_IMAGENET_INDICES,
    }
    config_path = os.path.join(ASSETS_DIR, "detector_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(detector_config, f, indent=2)
    config_kb = os.path.getsize(config_path) / 1024
    print(f"Config: {config_kb:.1f} KB  ({len(PLANT_IMAGENET_INDICES)} plant indices)")

    print(f"\nAssets written to: {ASSETS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
