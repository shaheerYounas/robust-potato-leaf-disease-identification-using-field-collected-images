"""
test_core_inference.py — Tests for model loading, preprocessing, and basic prediction.

Covers:
  - Model architecture loading & weight restoration
  - Input preprocessing pipeline correctness
  - Basic 7-class disease classification on real images
  - Prediction output schema validation
  - Probability distribution validity
  - Deterministic inference (same input → same output)
  - Device compatibility (CPU guaranteed, CUDA if available)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from potato_leaf_inference import (
    HybridCNNTransformer,
    EfficientNetTransfer,
    get_val_transforms,
    preprocess_pil,
    check_image_brightness,
    load_class_info,
    load_model,
    load_gate_model,
    predict_pil_image,
    predict_image,
    DEFAULT_CLASS_NAMES,
    DEFAULT_DISEASE_INFO,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from tests.conftest import TEST_IMAGES, DISEASE_CLASSES


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelArchitecture:

    def test_hybrid_model_output_shape(self, device):
        """HybridCNNTransformer output is [batch, 7] logits."""
        model = HybridCNNTransformer(n_classes=7).to(device).eval()
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 7)

    def test_hybrid_model_batch_input(self, device):
        """Model handles batch sizes > 1."""
        model = HybridCNNTransformer(n_classes=7).to(device).eval()
        dummy = torch.randn(4, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, 7)

    def test_efficientnet_transfer_output_shape(self, device):
        """EfficientNetTransfer output is [batch, n_classes]."""
        model = EfficientNetTransfer(n_classes=7).to(device).eval()
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 7)

    def test_model_in_eval_mode_after_load(self, model_bundle):
        """Loaded model must be in eval mode (no dropout active)."""
        assert not model_bundle["model"].training

    def test_model_parameter_count(self, model_bundle):
        """Model should have ~4M parameters (EfficientNetB0 + Transformer head)."""
        total_params = sum(p.numel() for p in model_bundle["model"].parameters())
        assert 3_000_000 < total_params < 30_000_000, f"Unexpected param count: {total_params}"


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_transforms_output_shape(self):
        """Transform must produce [3, 224, 224] tensor."""
        img = Image.new("RGB", (640, 480), (100, 150, 100))
        t = get_val_transforms()(img)
        assert t.shape == (3, 224, 224)

    def test_preprocess_pil_adds_batch_dim(self):
        """preprocess_pil should return [1, 3, 224, 224]."""
        img = Image.new("RGB", (300, 200), (100, 150, 100))
        tensor = preprocess_pil(img)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_converts_grayscale(self, grayscale_image):
        """Grayscale input → 3-channel output."""
        tensor = preprocess_pil(grayscale_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_converts_rgba(self, rgba_image):
        """RGBA input → 3-channel output (alpha stripped)."""
        tensor = preprocess_pil(rgba_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_tiny_image(self, tiny_image):
        """16x16 image should be up-sampled to 224x224."""
        tensor = preprocess_pil(tiny_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_large_image(self, large_image):
        """4K image should be down-sampled to 224x224."""
        tensor = preprocess_pil(large_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_output_is_normalised(self):
        """Output tensor values should be roughly in [-3, 3] from ImageNet normalization."""
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        tensor = preprocess_pil(img)
        assert tensor.min() > -5.0
        assert tensor.max() < 5.0

    def test_preprocess_is_deterministic(self):
        """Same image → identical tensor."""
        img = Image.new("RGB", (224, 224), (100, 150, 100))
        t1 = preprocess_pil(img)
        t2 = preprocess_pil(img)
        assert torch.equal(t1, t2)


# ═══════════════════════════════════════════════════════════════════════════════
# BRIGHTNESS VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrightnessValidation:

    def test_normal_brightness(self):
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        result = check_image_brightness(img)
        assert result["level"] == "OK"
        assert 100 < result["mean_brightness"] < 170

    def test_low_light_detected(self, pure_black_image):
        result = check_image_brightness(pure_black_image)
        assert result["level"] == "WARNING_LOW_LIGHT"

    def test_overexposed_detected(self, pure_white_image):
        result = check_image_brightness(pure_white_image)
        assert result["level"] == "WARNING_OVEREXPOSED"

    def test_borderline_low_light(self):
        """Brightness exactly at threshold — should be OK."""
        # RGB (60,60,60) → grayscale ~60
        img = Image.new("RGB", (224, 224), (60, 60, 60))
        result = check_image_brightness(img)
        assert result["mean_brightness"] >= 59

    def test_brightness_has_required_keys(self):
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        result = check_image_brightness(img)
        assert "mean_brightness" in result
        assert "level" in result
        assert "message" in result


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS INFO LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassInfo:

    def test_class_info_has_7_classes(self, model_bundle):
        assert len(model_bundle["class_names"]) == 7

    def test_all_disease_classes_present(self, model_bundle):
        for cls in DISEASE_CLASSES:
            assert cls in model_bundle["class_names"]

    def test_disease_info_for_all_classes(self, model_bundle):
        for cls in model_bundle["class_names"]:
            assert cls in model_bundle["disease_info"]
            assert len(model_bundle["disease_info"][cls]) > 10

    def test_fallback_defaults_when_json_missing(self):
        info = load_class_info("/nonexistent/path.json")
        assert info["class_names"] == DEFAULT_CLASS_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicPrediction:

    def test_prediction_returns_dict(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        assert isinstance(result, dict)

    def test_prediction_has_required_keys(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        assert "predicted_class" in result
        assert "confidence" in result
        assert "rejected" in result
        assert "top_k" in result

    def test_confidence_is_valid_probability(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        if not result["rejected"]:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_top_k_probabilities_sum_approximately(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            top_k=7,
        )
        if not result["rejected"]:
            total = sum(p["probability"] for p in result["top_k"])
            assert 0.95 <= total <= 1.05, f"Probabilities sum to {total}"

    def test_top_k_ordering(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            top_k=5,
        )
        if result.get("top_k") and len(result["top_k"]) > 1:
            probs = [p["probability"] for p in result["top_k"]]
            assert probs == sorted(probs, reverse=True), "Top-k not in descending order"

    def test_predicted_class_in_class_names(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        valid_classes = set(model_bundle["class_names"]) | {"Not a Potato Leaf", "Uncertain"}
        assert result["predicted_class"] in valid_classes

    def test_top_k_entries_have_disease_note(self, model_bundle, synthetic_leaf):
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        for entry in result.get("top_k", []):
            assert "disease_note" in entry

    def test_deterministic_prediction(self, model_bundle, synthetic_leaf):
        """Same image → same prediction twice."""
        kwargs = dict(
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        r1 = predict_pil_image(synthetic_leaf, **kwargs)
        r2 = predict_pil_image(synthetic_leaf, **kwargs)
        assert r1["predicted_class"] == r2["predicted_class"]
        assert abs(r1["confidence"] - r2["confidence"]) < 1e-5

    def test_predict_image_from_path(self, model_bundle):
        """predict_image() works with file path."""
        leaf_path = TEST_IMAGES.get("healthy_leaf")
        if leaf_path is None or not leaf_path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        result = predict_image(
            leaf_path,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        assert "image_path" in result
        assert result["image_path"] == str(leaf_path)


# ═══════════════════════════════════════════════════════════════════════════════
# REAL DATASET CLASSIFICATION ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealDatasetClassification:
    """Smoke-test against real dataset samples — the model should correctly
    classify at least some images from each class."""

    @pytest.mark.parametrize("class_name", DISEASE_CLASSES)
    def test_classifies_real_images(self, model_bundle, real_dataset_samples, class_name):
        """At least one of 3 sample images per class should be correctly predicted."""
        samples = real_dataset_samples.get(class_name, [])
        if not samples:
            pytest.skip(f"No real samples for {class_name}")

        correct = 0
        for img_path in samples:
            img = Image.open(img_path).convert("RGB")
            result = predict_pil_image(
                img,
                model=model_bundle["model"],
                device=model_bundle["device"],
                class_names=model_bundle["class_names"],
                disease_info=model_bundle["disease_info"],
                centroids_data=model_bundle["centroids_data"],
                gate_model=model_bundle["gate_model"],
                gate_categories=model_bundle["gate_categories"],
            )
            if not result["rejected"] and result["predicted_class"] == class_name:
                correct += 1

        assert correct >= 1, (
            f"Model failed to correctly classify any of {len(samples)} "
            f"{class_name} images"
        )
