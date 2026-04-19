# Deployment Guide - Potato Leaf Disease Classification

## Overview

The project has two active deployment targets:

- Python deployment from the repository root
- Android deployment packaged in `AdvancePractice/PotatoLeafDisease.apk`

The Python inference wrapper uses a 3-stage safety flow:

1. `leaf_gate`
   Rejects obvious non-leaf / non-vegetation inputs.
2. `imagenet_label`
   Adds a human-readable object hint for rejected or uncertain cases.
3. `confidence_gate`
   Returns `Uncertain` for ambiguous leaf-like images instead of forcing a disease label.

## Required Files

| File | Purpose |
|---|---|
| `artifacts/phase_2_benchmarking/models/hybrid_cnn_transformer_best.pt` | Final PyTorch checkpoint |
| `AdvancePractice/deployment/class_info.json` | Class metadata and gate configuration |
| `artifacts/phase_2_benchmarking/models/leaf_centroids.pt` | Leaf-gate centroids |
| `potato_leaf_inference.py` | Core inference pipeline |
| `predict.py` | CLI inference entry point |
| `app.py` | Streamlit web app |
| `explainability.py` | Optional Grad-CAM / Score-CAM visual explanations |

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# GPU example
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU-only example
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

## Run The Web App

```bash
streamlit run app.py
```

The web app supports:

- top-k disease predictions
- low-light and overexposure warnings
- non-leaf rejection feedback
- uncertain-image handling
- optional Grad-CAM / Score-CAM overlays for accepted predictions

## Run The CLI

```bash
python predict.py path/to/leaf_image.jpg
python predict.py path/to/leaf_image.jpg --top-k 5 --device cpu --json-out result.json
```

## Python Usage

```python
from PIL import Image

from potato_leaf_inference import load_gate_model, load_model, predict_pil_image

model, device, info = load_model()
gate_model, gate_categories = load_gate_model(device)

img = Image.open("leaf.jpg").convert("RGB")
result = predict_pil_image(
    img,
    model=model,
    device=device,
    class_names=info["class_names"],
    disease_info=info["disease_info"],
    centroids_data=info.get("_centroids"),
    gate_model=gate_model,
    gate_categories=gate_categories,
)
```

## Output Types

- Accepted disease prediction:
  Returns one of `Bacteria`, `Fungi`, `Healthy`, `Nematode`, `Pest`, `Phytopthora`, or `Virus`.
- Hard rejection:
  Returns `Not a Potato Leaf` with a rejection reason and detected object label.
- Soft rejection:
  Returns `Uncertain` with `best_guess`, confidence, and rejection reason.

## Model Details

- **Architecture:** Hybrid CNN-Transformer
- **Input:** 224 x 224 RGB, ImageNet-normalized
- **Output:** 7 disease classes plus rejection / uncertainty handling in the inference wrapper
- **Latency:** 1.339 ms/image (GPU), 35.85 ms/image (CPU)

## Extra Artifacts

- Final handoff package: `AdvancePractice/`
- Android APK: `AdvancePractice/PotatoLeafDisease.apk`
- ONNX export: `AdvancePractice/models/hybrid_cnn_transformer.onnx`
