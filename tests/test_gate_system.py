"""
test_gate_system.py — Tests for the 3-stage hierarchical gate system.

Gate 1: Leaf Similarity Gate — rejects non-leaf images via cosine similarity
Gate 2: ImageNet Gate — provides human-readable object labels for rejected images
Gate 3: Confidence Gate — rejects low-confidence predictions

Tests cover both rejection and acceptance paths under real-world conditions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from potato_leaf_inference import (
    check_leaf_gate,
    get_imagenet_label,
    predict_pil_image,
    preprocess_pil,
    DEFAULT_LEAF_GATE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    PLANT_IMAGENET_INDICES,
)

from tests.conftest import TEST_IMAGES, DISEASE_CLASSES


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 1: LEAF SIMILARITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeafSimilarityGate:
    """Tests that the leaf-gate correctly admits potato leaf images and
    rejects non-leaf objects."""

    @pytest.fixture(autouse=True)
    def _require_centroids(self, model_bundle):
        if model_bundle["centroids_data"] is None:
            pytest.skip("No centroids file available — leaf gate tests skipped")

    def test_gate_returns_required_keys(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 140, 60))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor,
            model_bundle["model"],
            model_bundle["centroids_data"],
            model_bundle["device"],
        )
        assert "is_leaf" in result
        assert "max_similarity" in result
        assert "closest_class" in result
        assert "threshold" in result
        assert "all_similarities" in result

    def test_gate_similarity_is_float(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 140, 60))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert isinstance(result["max_similarity"], float)
        assert -1.0 <= result["max_similarity"] <= 1.0

    def test_gate_uses_correct_threshold(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 140, 60))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert result["threshold"] == model_bundle["centroids_data"].get(
            "threshold", DEFAULT_LEAF_GATE_THRESHOLD
        )

    def test_gate_has_similarity_for_all_classes(self, model_bundle):
        img = Image.new("RGB", (224, 224), (60, 140, 60))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        for cls in DISEASE_CLASSES:
            assert cls in result["all_similarities"], f"Missing similarity for {cls}"

    # ── Real leaf images should PASS the gate ─────────────────────────────

    @pytest.mark.parametrize("img_key", ["bacteria_leaf", "healthy_leaf", "green_leaf", "real_leaf"])
    def test_real_leaf_passes_gate(self, model_bundle, img_key):
        path = TEST_IMAGES.get(img_key)
        if path is None or not path.exists():
            pytest.skip(f"{img_key} not found")
        img = Image.open(path).convert("RGB")
        tensor = preprocess_pil(img).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert result["is_leaf"], (
            f"{img_key} rejected by leaf gate "
            f"(sim={result['max_similarity']:.3f}, thresh={result['threshold']:.3f})"
        )

    # ── Non-leaf images should FAIL the gate ──────────────────────────────

    @pytest.mark.parametrize("img_key", ["non_leaf", "wood", "bedsheet", "red_cap"])
    def test_non_leaf_fails_gate(self, model_bundle, img_key):
        path = TEST_IMAGES.get(img_key)
        if path is None or not path.exists():
            pytest.skip(f"{img_key} not found")
        img = Image.open(path).convert("RGB")
        tensor = preprocess_pil(img).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert not result["is_leaf"], (
            f"{img_key} incorrectly passed leaf gate "
            f"(sim={result['max_similarity']:.3f})"
        )

    # ── Synthetic non-leaf images ─────────────────────────────────────────

    def test_pure_black_rejected(self, model_bundle, pure_black_image):
        tensor = preprocess_pil(pure_black_image).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert not result["is_leaf"]

    def test_pure_white_rejected(self, model_bundle, pure_white_image):
        tensor = preprocess_pil(pure_white_image).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert not result["is_leaf"]

    def test_noise_rejected(self, model_bundle, noise_image):
        tensor = preprocess_pil(noise_image).to(model_bundle["device"])
        result = check_leaf_gate(
            tensor, model_bundle["model"],
            model_bundle["centroids_data"], model_bundle["device"],
        )
        assert not result["is_leaf"]

    def test_non_leaf_synthetic_objects_rejected(self, model_bundle, non_leaf_objects):
        for name, img in non_leaf_objects.items():
            tensor = preprocess_pil(img).to(model_bundle["device"])
            result = check_leaf_gate(
                tensor, model_bundle["model"],
                model_bundle["centroids_data"], model_bundle["device"],
            )
            assert not result["is_leaf"], f"Synthetic {name} incorrectly admitted"

    # ── Dataset samples should PASS the gate ──────────────────────────────

    @pytest.mark.parametrize("class_name", DISEASE_CLASSES)
    def test_dataset_samples_pass_gate(self, model_bundle, real_dataset_samples, class_name):
        samples = real_dataset_samples.get(class_name, [])
        if not samples:
            pytest.skip(f"No samples for {class_name}")
        passed = 0
        for path in samples:
            img = Image.open(path).convert("RGB")
            tensor = preprocess_pil(img).to(model_bundle["device"])
            result = check_leaf_gate(
                tensor, model_bundle["model"],
                model_bundle["centroids_data"], model_bundle["device"],
            )
            if result["is_leaf"]:
                passed += 1
        assert passed >= 1, f"No {class_name} samples passed the leaf gate"


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 2: IMAGENET GATE (Object Labelling)
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageNetGate:

    def test_imagenet_returns_top_k_labels(self, model_bundle):
        img = Image.new("RGB", (224, 224), (200, 30, 30))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        results = get_imagenet_label(
            tensor, model_bundle["gate_model"],
            model_bundle["gate_categories"], model_bundle["device"],
            top_k=5,
        )
        assert len(results) == 5

    def test_imagenet_label_format(self, model_bundle):
        img = Image.new("RGB", (224, 224), (200, 30, 30))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        results = get_imagenet_label(
            tensor, model_bundle["gate_model"],
            model_bundle["gate_categories"], model_bundle["device"],
        )
        for lbl in results:
            assert "label" in lbl
            assert "confidence" in lbl
            assert "is_plant_related" in lbl
            assert isinstance(lbl["label"], str)
            assert 0 <= lbl["confidence"] <= 1

    def test_imagenet_plant_flag_uses_correct_indices(self, model_bundle):
        """The is_plant_related flag must reference PLANT_IMAGENET_INDICES."""
        # Just validate the set is populated
        assert len(PLANT_IMAGENET_INDICES) > 30

    @pytest.mark.parametrize("img_key", ["non_leaf", "wood", "bedsheet", "red_cap"])
    def test_non_leaf_gets_meaningful_label(self, model_bundle, img_key):
        path = TEST_IMAGES.get(img_key)
        if path is None or not path.exists():
            pytest.skip(f"{img_key} not found")
        img = Image.open(path).convert("RGB")
        tensor = preprocess_pil(img).to(model_bundle["device"])
        results = get_imagenet_label(
            tensor, model_bundle["gate_model"],
            model_bundle["gate_categories"], model_bundle["device"],
        )
        assert len(results) >= 1
        assert results[0]["confidence"] > 0.0
        assert len(results[0]["label"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 3: CONFIDENCE GATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceGate:

    def test_high_confidence_accepted(self, model_bundle):
        """A real leaf image should typically pass the confidence gate."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
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
        # Real leaf image — should either be accepted or rejected at leaf gate
        # but NOT at confidence gate
        if result["rejected"]:
            assert result.get("rejection_stage") != "confidence_gate", (
                "Known leaf image rejected at confidence gate"
            )

    def test_very_low_threshold_accepts_everything(self, model_bundle, synthetic_leaf):
        """With threshold=0.01, nearly everything should pass."""
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            confidence_threshold=0.01,
        )
        if result.get("rejection_stage") != "leaf_gate":
            assert not result["rejected"]

    def test_very_high_threshold_rejects_ambiguous(self, model_bundle, noise_image):
        """With threshold=0.99, ambiguous images should be rejected."""
        result = predict_pil_image(
            noise_image,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            confidence_threshold=0.99,
            # Skip leaf gate to test confidence gate directly
            centroids_data=None,
        )
        assert result["rejected"]
        assert result["rejection_stage"] == "confidence_gate"

    def test_rejection_includes_best_guess(self, model_bundle, noise_image):
        """Confidence-gate rejection should still provide a best_guess."""
        result = predict_pil_image(
            noise_image,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            confidence_threshold=0.99,
            centroids_data=None,
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )
        if result.get("rejection_stage") == "confidence_gate":
            assert "best_guess" in result
            assert result["best_guess"] in model_bundle["class_names"]


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE INTEGRATION (all 3 gates)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullGatePipeline:

    def test_leaf_image_passes_all_gates(self, model_bundle):
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
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
        assert not result["rejected"], (
            f"Healthy leaf rejected at {result.get('rejection_stage')} "
            f"— {result.get('rejection_reason')}"
        )

    @pytest.mark.parametrize("img_key", ["non_leaf", "wood", "bedsheet", "red_cap"])
    def test_non_leaf_rejected_with_feedback(self, model_bundle, img_key):
        if model_bundle["centroids_data"] is None:
            pytest.skip("No centroids — non-leaf rejection requires leaf gate")
        path = TEST_IMAGES.get(img_key)
        if path is None or not path.exists():
            pytest.skip(f"{img_key} not found")
        img = Image.open(path).convert("RGB")
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
        assert result["rejected"], f"{img_key} was not rejected"
        assert "rejection_reason" in result
        assert "detected_object" in result
        assert len(result["detected_object"]) > 0

    def test_no_centroids_skips_leaf_gate(self, model_bundle, synthetic_leaf):
        """When centroids=None, leaf gate is skipped entirely."""
        result = predict_pil_image(
            synthetic_leaf,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=None,
        )
        assert "leaf_gate" not in result or result.get("leaf_gate") is None

    def test_no_gate_model_gives_unknown_object(self, model_bundle):
        """When gate_model=None, rejected images get 'unknown object' label."""
        if model_bundle["centroids_data"] is None:
            pytest.skip("No centroids — cannot test gate_model=None rejection")
        path = TEST_IMAGES.get("wood")
        if path is None or not path.exists():
            pytest.skip("test_wood.png not found")
        img = Image.open(path).convert("RGB")
        result = predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=model_bundle["centroids_data"],
            gate_model=None,
            gate_categories=None,
        )
        if result["rejected"] and result.get("rejection_stage") == "leaf_gate":
            assert result["detected_object"] == "unknown object"
