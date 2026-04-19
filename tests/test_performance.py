"""
test_performance.py — Performance and latency benchmarks for the potato leaf
disease inference pipeline.

Covers:
  - Single-image inference latency (CPU)
  - Batch throughput (multiple images in sequence)
  - Model loading time
  - Memory footprint estimation
  - Preprocessing latency
  - Gate system overhead
  - Consistency under repeated inference
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from potato_leaf_inference import (
    load_model,
    load_gate_model,
    predict_pil_image,
    preprocess_pil,
    check_image_brightness,
    check_leaf_gate,
    get_imagenet_label,
    get_val_transforms,
)

from tests.conftest import TEST_IMAGES, PROJECT_ROOT


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _time_fn(fn, *args, repeats=5, **kwargs):
    """Run fn() `repeats` times, return (median_seconds, all_times)."""
    times = []
    result = None
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times)), times, result


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelLoadingPerformance:

    def test_model_loading_time(self):
        """Full load_model() should complete within 30 s on CPU."""
        median, times, _ = _time_fn(load_model, device="cpu", repeats=3)
        assert median < 30.0, f"Model loading too slow: {median:.2f}s"

    def test_gate_model_loading_time(self):
        """Gate model should load within 20 s."""
        device = torch.device("cpu")
        median, times, _ = _time_fn(load_gate_model, device, repeats=3)
        assert median < 20.0, f"Gate model loading too slow: {median:.2f}s"

    def test_model_parameter_memory(self, model_bundle):
        """Estimate total parameter memory (float32)."""
        model = model_bundle["model"]
        total_params = sum(p.numel() for p in model.parameters())
        param_bytes = total_params * 4  # float32
        param_mb = param_bytes / (1024 ** 2)
        # Model should be under 150 MB in float32
        assert param_mb < 150, f"Model too large: {param_mb:.1f} MB"


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessingPerformance:

    def test_preprocess_latency(self):
        """Single preprocess call should take < 50 ms."""
        img = Image.new("RGB", (640, 480), (80, 140, 80))
        median, _, _ = _time_fn(preprocess_pil, img, repeats=20)
        assert median < 0.05, f"Preprocessing too slow: {median*1000:.1f} ms"

    def test_preprocess_large_image(self):
        """Preprocessing a 4K image should take < 200 ms."""
        img = Image.new("RGB", (3840, 2160), (80, 140, 80))
        median, _, _ = _time_fn(preprocess_pil, img, repeats=10)
        assert median < 0.2, f"4K preprocessing too slow: {median*1000:.1f} ms"

    def test_brightness_check_latency(self):
        """Brightness validation should take < 20 ms."""
        img = Image.new("RGB", (640, 480), (120, 120, 120))
        median, _, _ = _time_fn(check_image_brightness, img, repeats=20)
        assert median < 0.02, f"Brightness check too slow: {median*1000:.1f} ms"


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-IMAGE INFERENCE LATENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleImageLatency:

    def _predict(self, img, bundle):
        return predict_pil_image(
            img,
            model=bundle["model"],
            device=bundle["device"],
            class_names=bundle["class_names"],
            disease_info=bundle["disease_info"],
            centroids_data=bundle["centroids_data"],
            gate_model=bundle["gate_model"],
            gate_categories=bundle["gate_categories"],
        )

    def test_synthetic_image_latency(self, model_bundle):
        """Inference on a synthetic image should take < 5 s on CPU."""
        img = Image.new("RGB", (224, 224), (50, 130, 50))
        median, _, _ = _time_fn(self._predict, img, model_bundle, repeats=5)
        assert median < 5.0, f"Inference too slow: {median:.2f}s"

    def test_real_leaf_latency(self, model_bundle):
        """Inference on a real leaf image should take < 5 s on CPU."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        median, _, _ = _time_fn(self._predict, img, model_bundle, repeats=5)
        assert median < 5.0, f"Real-leaf inference too slow: {median:.2f}s"

    def test_non_leaf_rejection_latency(self, model_bundle):
        """Rejected (non-leaf) images should still complete < 5 s."""
        path = TEST_IMAGES.get("wood")
        if path is None or not path.exists():
            pytest.skip("test_wood.png not found")
        img = Image.open(path).convert("RGB")
        median, _, _ = _time_fn(self._predict, img, model_bundle, repeats=5)
        assert median < 5.0, f"Rejection too slow: {median:.2f}s"

    def test_large_image_latency(self, model_bundle):
        """4K image should not explode inference time (< 10 s)."""
        img = Image.new("RGB", (3840, 2160), (60, 130, 60))
        median, _, _ = _time_fn(self._predict, img, model_bundle, repeats=3)
        assert median < 10.0, f"4K inference too slow: {median:.2f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH THROUGHPUT
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchThroughput:

    def test_sequential_throughput_20_images(self, model_bundle):
        """Process 20 synthetic images and measure throughput (img/s)."""
        imgs = [Image.new("RGB", (224, 224), (40 + i * 5, 130, 40 + i * 5))
                for i in range(20)]

        t0 = time.perf_counter()
        for img in imgs:
            predict_pil_image(
                img,
                model=model_bundle["model"],
                device=model_bundle["device"],
                class_names=model_bundle["class_names"],
                disease_info=model_bundle["disease_info"],
            )
        elapsed = time.perf_counter() - t0
        throughput = 20 / elapsed
        # At least 1 img/s on CPU
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} img/s"

    def test_mixed_input_throughput(self, model_bundle):
        """Mix of leaf and non-leaf images — both paths exercised."""
        images = []
        for name, path in TEST_IMAGES.items():
            if path.exists():
                images.append(Image.open(path).convert("RGB"))
        if len(images) < 2:
            pytest.skip("Not enough test images")

        t0 = time.perf_counter()
        for img in images:
            predict_pil_image(
                img,
                model=model_bundle["model"],
                device=model_bundle["device"],
                class_names=model_bundle["class_names"],
                disease_info=model_bundle["disease_info"],
                centroids_data=model_bundle["centroids_data"],
                gate_model=model_bundle["gate_model"],
                gate_categories=model_bundle["gate_categories"],
            )
        elapsed = time.perf_counter() - t0
        throughput = len(images) / elapsed
        assert throughput > 0.5, f"Mixed throughput too low: {throughput:.2f} img/s"


# ═══════════════════════════════════════════════════════════════════════════════
# GATE SYSTEM OVERHEAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateOverhead:

    def test_leaf_gate_latency(self, model_bundle):
        """Leaf similarity gate should add < 1 s overhead."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        centroids = model_bundle["centroids_data"]
        if centroids is None:
            pytest.skip("No centroids available")

        median, _, _ = _time_fn(
            check_leaf_gate,
            tensor, model_bundle["model"], centroids, model_bundle["device"],
            repeats=10,
        )
        assert median < 1.0, f"Leaf gate too slow: {median*1000:.1f} ms"

    def test_imagenet_gate_latency(self, model_bundle):
        """ImageNet label gate should complete < 3 s."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        tensor = preprocess_pil(img).to(model_bundle["device"])
        gate_model = model_bundle["gate_model"]
        gate_cats = model_bundle["gate_categories"]
        if gate_model is None:
            pytest.skip("No gate model")

        median, _, _ = _time_fn(
            get_imagenet_label,
            tensor, gate_model, gate_cats, model_bundle["device"],
            repeats=5,
        )
        assert median < 3.0, f"ImageNet gate too slow: {median:.2f}s"

    def test_full_pipeline_vs_no_gates(self, model_bundle):
        """Full pipeline (with gates) should be ≤ 3× slower than without."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        kwargs_no_gates = dict(
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )
        kwargs_full = dict(
            **kwargs_no_gates,
            centroids_data=model_bundle["centroids_data"],
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )

        med_no, _, _ = _time_fn(predict_pil_image, img, **kwargs_no_gates, repeats=5)
        med_full, _, _ = _time_fn(predict_pil_image, img, **kwargs_full, repeats=5)

        ratio = med_full / max(med_no, 1e-6)
        assert ratio < 3.0, f"Gates add too much overhead: {ratio:.1f}x"


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM & STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceStability:

    def test_deterministic_over_10_runs(self, model_bundle):
        """Same image → same prediction 10 times."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        results = []
        for _ in range(10):
            r = predict_pil_image(
                img,
                model=model_bundle["model"],
                device=model_bundle["device"],
                class_names=model_bundle["class_names"],
                disease_info=model_bundle["disease_info"],
            )
            results.append((r["predicted_class"], r["confidence"]))

        classes = {c for c, _ in results}
        confs = [c for _, c in results]
        assert len(classes) == 1, f"Non-deterministic classes: {classes}"
        assert max(confs) - min(confs) < 1e-5, "Confidence varies across runs"

    def test_confidence_variance_across_inputs(self, model_bundle):
        """Different images should produce different confidences (model isn't collapsed)."""
        confs = []
        for name, path in TEST_IMAGES.items():
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            r = predict_pil_image(
                img,
                model=model_bundle["model"],
                device=model_bundle["device"],
                class_names=model_bundle["class_names"],
                disease_info=model_bundle["disease_info"],
            )
            confs.append(r["confidence"])
        if len(confs) < 3:
            pytest.skip("Need at least 3 test images")
        assert np.std(confs) > 0.01, "All confidences identical — model may be collapsed"
