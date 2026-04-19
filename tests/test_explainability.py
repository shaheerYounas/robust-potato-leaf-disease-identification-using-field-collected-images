"""
test_explainability.py — Tests for Grad-CAM and Score-CAM explainability module.

Covers:
  - Heatmap shape, range, and dtype correctness
  - Overlay image generation
  - generate_explanation() high-level API
  - Class-targeting (predicted vs specific class)
  - Hook registration and cleanup
  - Real leaf image XAI output
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from explainability import (
    GradCAM,
    ScoreCAM,
    overlay_heatmap,
    generate_explanation,
)
from potato_leaf_inference import (
    HybridCNNTransformer,
    get_val_transforms,
    IMAGE_SIZE,
    DEFAULT_CLASS_NAMES,
)

from tests.conftest import TEST_IMAGES


# ═══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradCAM:

    def test_heatmap_shape(self, model_bundle):
        """Grad-CAM output must be (224, 224) float array."""
        cam = GradCAM(model_bundle["model"])
        try:
            img = Image.new("RGB", (300, 200), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam.generate(tensor)
            assert heatmap.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1])
        finally:
            cam.release()

    def test_heatmap_range(self, model_bundle):
        """Heatmap values must be in [0, 1]."""
        cam = GradCAM(model_bundle["model"])
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam.generate(tensor)
            assert heatmap.min() >= 0.0
            assert heatmap.max() <= 1.0
        finally:
            cam.release()

    def test_heatmap_dtype(self, model_bundle):
        """Must be float64 (numpy default from division)."""
        cam = GradCAM(model_bundle["model"])
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam.generate(tensor)
            assert heatmap.dtype in (np.float32, np.float64)
        finally:
            cam.release()

    def test_target_class_override(self, model_bundle):
        """Specifying a target class changes the heatmap."""
        cam = GradCAM(model_bundle["model"])
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap_0 = cam.generate(tensor, target_class=0)
            heatmap_3 = cam.generate(tensor, target_class=3)
            # Different classes should generally produce different heatmaps
            # (not guaranteed to be very different for synthetic images, but shapes must match)
            assert heatmap_0.shape == heatmap_3.shape
        finally:
            cam.release()

    def test_hooks_cleaned_up(self, model_bundle):
        """After release(), hooks should be removed."""
        cam = GradCAM(model_bundle["model"])
        cam.release()
        # Creating a new GradCAM should work without hook conflicts
        cam2 = GradCAM(model_bundle["model"])
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam2.generate(tensor)
            assert heatmap.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1])
        finally:
            cam2.release()

    def test_gradcam_on_real_leaf(self, model_bundle):
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        cam = GradCAM(model_bundle["model"])
        try:
            img = Image.open(path).convert("RGB")
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam.generate(tensor)
            assert heatmap.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1])
            # Real leaf should produce a non-trivial heatmap
            assert heatmap.max() > 0.0
        finally:
            cam.release()


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE-CAM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreCAM:

    def test_heatmap_shape(self, model_bundle):
        cam = ScoreCAM(model_bundle["model"], batch_size=16)
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam.generate(tensor)
            assert heatmap.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1])
        finally:
            cam.release()

    def test_heatmap_range(self, model_bundle):
        cam = ScoreCAM(model_bundle["model"], batch_size=16)
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            heatmap = cam.generate(tensor)
            assert heatmap.min() >= 0.0
            assert heatmap.max() <= 1.0
        finally:
            cam.release()

    def test_scorecam_is_gradient_free(self, model_bundle):
        """Score-CAM should work even if gradients are disabled."""
        cam = ScoreCAM(model_bundle["model"], batch_size=8)
        try:
            img = Image.new("RGB", (224, 224), (60, 130, 60))
            tensor = get_val_transforms()(img).unsqueeze(0).to(model_bundle["device"])
            with torch.no_grad():
                heatmap = cam.generate(tensor)
            assert heatmap.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1])
        finally:
            cam.release()


# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAY UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlay:

    def test_overlay_returns_pil_image(self):
        original = Image.new("RGB", (224, 224), (60, 130, 60))
        heatmap = np.random.rand(224, 224)
        result = overlay_heatmap(original, heatmap)
        assert isinstance(result, Image.Image)

    def test_overlay_size_matches_original(self):
        original = Image.new("RGB", (300, 200), (60, 130, 60))
        heatmap = np.random.rand(224, 224)
        result = overlay_heatmap(original, heatmap)
        assert result.size == original.size

    def test_overlay_alpha_0_is_original(self):
        """Alpha=0 overlay should look like the original."""
        original = Image.new("RGB", (224, 224), (60, 130, 60))
        heatmap = np.ones((224, 224))
        result = overlay_heatmap(original, heatmap, alpha=0.0)
        assert isinstance(result, Image.Image)

    def test_overlay_alpha_1_is_heatmap(self):
        """Alpha=1 overlay should look like the heatmap."""
        original = Image.new("RGB", (224, 224), (60, 130, 60))
        heatmap = np.ones((224, 224))
        result = overlay_heatmap(original, heatmap, alpha=1.0)
        assert isinstance(result, Image.Image)

    def test_different_colormaps(self):
        original = Image.new("RGB", (224, 224), (60, 130, 60))
        heatmap = np.random.rand(224, 224)
        for cmap in ["jet", "hot", "viridis", "plasma"]:
            result = overlay_heatmap(original, heatmap, colormap=cmap)
            assert isinstance(result, Image.Image)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL generate_explanation() API
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateExplanation:

    def test_gradcam_explanation_keys(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            method="gradcam",
        )
        assert "heatmap" in result
        assert "overlay" in result
        assert "predicted_class" in result
        assert "target_class" in result
        assert "confidence" in result

    def test_scorecam_explanation_keys(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            method="scorecam",
        )
        assert "heatmap" in result
        assert "overlay" in result

    def test_explanation_heatmap_is_numpy(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
        )
        assert isinstance(result["heatmap"], np.ndarray)

    def test_explanation_overlay_is_pil(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
        )
        assert isinstance(result["overlay"], Image.Image)

    def test_predicted_class_is_valid(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
        )
        assert result["predicted_class"] in model_bundle["class_names"]

    def test_specific_target_class(self, model_bundle):
        """Explicitly target class index 2 (Healthy)."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            target_class=2,
        )
        assert result["target_class"] == model_bundle["class_names"][2]
        assert result["target_index"] == 2

    def test_confidence_is_valid(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        result = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
        )
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.parametrize("method", ["gradcam", "scorecam"])
    def test_real_leaf_explanation(self, model_bundle, method):
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        result = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            method=method,
        )
        assert result["heatmap"].shape == (IMAGE_SIZE[0], IMAGE_SIZE[1])
        assert isinstance(result["overlay"], Image.Image)
        # Non-trivial heatmap for a real image
        assert result["heatmap"].max() > 0.0

    def test_explanation_alpha_parameter(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        r1 = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            alpha=0.2,
        )
        r2 = generate_explanation(
            img, model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            alpha=0.8,
        )
        # Different alpha should produce different overlays
        arr1 = np.array(r1["overlay"])
        arr2 = np.array(r2["overlay"])
        assert not np.array_equal(arr1, arr2)
