"""
conftest.py — Shared pytest fixtures for the potato leaf disease test suite.

Loads models once per session to avoid repeated I/O across hundreds of tests.
Provides synthetic and real image factories for every test scenario.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFilter

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Real-world test images bundled with the project ───────────────────────────
TEST_IMAGES = {
    "bacteria_leaf": PROJECT_ROOT / "test_bacteria_leaf.jpg",
    "healthy_leaf": PROJECT_ROOT / "test_healthy_leaf.jpg",
    "green_leaf": PROJECT_ROOT / "test_green_leaf.jpg",
    "real_leaf": PROJECT_ROOT / "test_real_leaf.png",
    "non_leaf": PROJECT_ROOT / "test_non_leaf.jpg",
    "wood": PROJECT_ROOT / "test_wood.png",
    "bedsheet": PROJECT_ROOT / "test_bedsheet.png",
    "red_cap": PROJECT_ROOT / "test_red_cap.png",
}

# ── Dataset path ──────────────────────────────────────────────────────────────
DATASET_ROOT = (
    PROJECT_ROOT / "data" / "raw"
    / "Potato Leaf Disease Dataset in Uncontrolled Environment"
    / "Potato Leaf Disease Dataset in Uncontrolled Environment"
)
DISEASE_CLASSES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION-SCOPED MODEL FIXTURES  (loaded once, reused everywhere)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model_bundle(device):
    """Load the full inference stack: model + gate model + centroids."""
    from potato_leaf_inference import load_model, load_gate_model

    model, active_device, info = load_model(device=str(device))
    gate_model, gate_categories = load_gate_model(active_device)
    centroids_data = info.get("_centroids")
    return {
        "model": model,
        "device": active_device,
        "info": info,
        "gate_model": gate_model,
        "gate_categories": gate_categories,
        "centroids_data": centroids_data,
        "class_names": info["class_names"],
        "disease_info": info["disease_info"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_leaf():
    """Generate a synthetic green leaf-like image (solid green with veins)."""
    img = Image.new("RGB", (640, 480), (40, 140, 40))
    draw = ImageDraw.Draw(img)
    # Main vein
    draw.line([(320, 30), (320, 450)], fill=(30, 100, 30), width=4)
    # Side veins
    for y in range(80, 420, 50):
        draw.line([(320, y), (200, y - 30)], fill=(30, 100, 30), width=2)
        draw.line([(320, y), (440, y - 30)], fill=(30, 100, 30), width=2)
    return img


@pytest.fixture
def pure_black_image():
    return Image.new("RGB", (224, 224), (0, 0, 0))


@pytest.fixture
def pure_white_image():
    return Image.new("RGB", (224, 224), (255, 255, 255))


@pytest.fixture
def noise_image():
    """Random noise — no structure at all."""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def tiny_image():
    """Very small image (16x16) — tests resize robustness."""
    return Image.new("RGB", (16, 16), (60, 130, 60))


@pytest.fixture
def large_image():
    """4K-resolution image — tests memory handling."""
    return Image.new("RGB", (3840, 2160), (90, 160, 70))


@pytest.fixture
def grayscale_image():
    """Single-channel grayscale image that must be converted."""
    return Image.new("L", (224, 224), 128)


@pytest.fixture
def rgba_image():
    """RGBA image with alpha channel — tests .convert('RGB') path."""
    img = Image.new("RGBA", (224, 224), (60, 130, 60, 200))
    return img


@pytest.fixture
def low_light_leaf(synthetic_leaf):
    """Extremely dark leaf (brightness ~20)."""
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(synthetic_leaf).enhance(0.08)


@pytest.fixture
def overexposed_leaf(synthetic_leaf):
    """Blown-out highlights (mean brightness > 220)."""
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(synthetic_leaf).enhance(6.0)


@pytest.fixture
def blurred_leaf(synthetic_leaf):
    """Heavy Gaussian blur — simulates out-of-focus field capture."""
    return synthetic_leaf.filter(ImageFilter.GaussianBlur(radius=12))


@pytest.fixture
def rotated_leaf(synthetic_leaf):
    """90-degree rotated leaf — tests orientation robustness."""
    return synthetic_leaf.rotate(90, expand=True)


@pytest.fixture
def cropped_partial_leaf(synthetic_leaf):
    """Only bottom-right quarter — partial occlusion scenario."""
    w, h = synthetic_leaf.size
    return synthetic_leaf.crop((w // 2, h // 2, w, h))


@pytest.fixture
def high_contrast_leaf(synthetic_leaf):
    """Extreme contrast enhancement."""
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(synthetic_leaf).enhance(4.0)


@pytest.fixture
def salt_pepper_image():
    """Salt-and-pepper noise overlaid on a green base."""
    arr = np.full((224, 224, 3), (60, 130, 60), dtype=np.uint8)
    mask = np.random.random((224, 224))
    arr[mask < 0.05] = 0
    arr[mask > 0.95] = 255
    return Image.fromarray(arr)


@pytest.fixture
def jpeg_artifact_leaf(synthetic_leaf, tmp_path):
    """Heavily JPEG-compressed leaf — quality=5."""
    path = tmp_path / "low_quality.jpg"
    synthetic_leaf.save(str(path), "JPEG", quality=5)
    return Image.open(str(path)).copy()


@pytest.fixture
def non_leaf_objects():
    """Return dict of non-leaf synthetic images."""
    images = {}
    # Red square
    images["red_square"] = Image.new("RGB", (224, 224), (200, 30, 30))
    # Blue gradient (sky-like)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        arr[i, :] = (100 + int(i * 0.5), 150 + int(i * 0.3), 220)
    images["blue_gradient"] = Image.fromarray(arr)
    # Brown (soil)
    images["soil"] = Image.new("RGB", (224, 224), (139, 90, 43))
    return images


# ═══════════════════════════════════════════════════════════════════════════════
# REAL DATASET SAMPLE FIXTURE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_dataset_samples(class_name: str, max_samples: int = 3) -> list[Path]:
    """Return up to max_samples real images from a dataset class."""
    class_dir = DATASET_ROOT / class_name
    if not class_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in class_dir.iterdir() if p.suffix.lower() in exts])[:max_samples]


@pytest.fixture(scope="session")
def real_dataset_samples():
    """Dict mapping class name → list of Path for real dataset images."""
    samples = {}
    for cls in DISEASE_CLASSES:
        paths = _get_dataset_samples(cls, max_samples=3)
        if paths:
            samples[cls] = paths
    return samples


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory cleaned up after test."""
    out = tmp_path / "test_output"
    out.mkdir()
    return out
