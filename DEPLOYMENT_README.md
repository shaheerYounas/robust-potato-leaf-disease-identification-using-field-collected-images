# Deployment Guide — Potato Leaf Disease Classification

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 (adjust for your GPU)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU-only alternative
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Required Files

| File | Purpose |
|---|---|
| `artifacts/phase_2_benchmarking/models/efficientnet_b0_finetune_best.pt` | Model weights (~32 MB) |
| `submission_ready/final_package/deployment/class_info.json` | Class names + disease descriptions |
| `potato_leaf_inference.py` | Core inference module |

### 3. Run the Web App (Streamlit)

```bash
streamlit run app.py
```

- Upload a JPG/PNG/BMP/WebP leaf image
- Displays top-k predictions with confidence scores and disease descriptions
- Automatically warns on low-light or overexposed images

### 4. Run CLI Inference

```bash
# Basic usage
python predict.py path/to/leaf_image.jpg

# With options
python predict.py path/to/leaf_image.jpg --top-k 5 --device cuda --json-out result.json
```

### 5. Use as a Python Module

```python
from potato_leaf_inference import load_model, predict_image, predict_pil_image, check_image_brightness
from PIL import Image

# Load model (auto-detects GPU)
model, device, info = load_model()

# Predict from file path
result = predict_image("leaf.jpg", model, device, info["class_names"], info["disease_info"])

# Predict from PIL image
img = Image.open("leaf.jpg")
result = predict_pil_image(img, model, device, info["class_names"], info["disease_info"])

# Check brightness before inference
brightness = check_image_brightness(img)
if brightness["level"] != "OK":
    print(f"Warning: {brightness['message']}")
```

## Model Details

- **Architecture:** EfficientNetB0 (fine-tuned last 2 blocks) + custom classifier head
- **Input:** 224 × 224 RGB, ImageNet-normalised
- **Output:** 7-class softmax (Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus)
- **Latency:** ~1.3 ms/image (GPU), ~82 ms/image (CPU)

## Known Limitations

- **Low-light images:** 9.91% F1 drop under dimmed conditions — brightness validation warns users automatically
- **Single dataset:** Trained on Mendeley Potato Leaf Disease dataset only; cross-dataset generalisation untested
- **No mobile optimisation:** ONNX export available but INT8 quantisation not verified
