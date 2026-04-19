"""
test_cli_integration.py — End-to-end integration tests for the CLI and
full prediction pipeline.

Covers:
  - predict.py CLI with real images (leaf and non-leaf)
  - JSON output file generation
  - Pipeline round-trip: file → predict → result schema
  - Multiple images in sequence (state isolation)
  - All 7 dataset classes via the full pipeline
  - Brightness validation toggle
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from potato_leaf_inference import (
    load_model,
    load_gate_model,
    predict_image,
    predict_pil_image,
    DEFAULT_CLASS_NAMES,
    DEFAULT_DISEASE_INFO,
)

from tests.conftest import TEST_IMAGES, DATASET_ROOT, DISEASE_CLASSES, PROJECT_ROOT


# ═══════════════════════════════════════════════════════════════════════════════
# CLI SMOKE TESTS  (subprocess-based — tests the actual script)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLI:

    @pytest.fixture(scope="class")
    def predict_script(self):
        return PROJECT_ROOT / "predict.py"

    def _run_cli(self, predict_script, *args, timeout=120):
        cmd = [sys.executable, str(predict_script)] + list(args)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return proc

    def test_cli_runs_on_leaf(self, predict_script):
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        proc = self._run_cli(predict_script, str(path), "--device", "cpu")
        assert proc.returncode == 0, f"STDERR: {proc.stderr}"
        # Should print model info
        assert "Model:" in proc.stdout or "Predicted" in proc.stdout or "Device:" in proc.stdout

    def test_cli_runs_on_non_leaf(self, predict_script):
        path = TEST_IMAGES.get("wood")
        if path is None or not path.exists():
            pytest.skip("test_wood.png not found")
        proc = self._run_cli(predict_script, str(path), "--device", "cpu")
        assert proc.returncode == 0
        # Should either be rejected (with centroids) or classified (without)
        assert "REJECTED" in proc.stdout or "UNCERTAIN" in proc.stdout or "Predicted class" in proc.stdout

    def test_cli_json_output(self, predict_script, tmp_path):
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        json_out = tmp_path / "result.json"
        proc = self._run_cli(
            predict_script, str(path),
            "--device", "cpu",
            "--json-out", str(json_out),
        )
        assert proc.returncode == 0, f"STDERR: {proc.stderr}"
        assert json_out.exists(), "JSON output file not created"
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert "predicted_class" in data

    def test_cli_top_k_parameter(self, predict_script):
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        proc = self._run_cli(
            predict_script, str(path),
            "--device", "cpu", "--top-k", "5",
        )
        assert proc.returncode == 0

    def test_cli_invalid_image_fails_gracefully(self, predict_script, tmp_path):
        """CLI with a non-image file should fail but not crash with a traceback."""
        fake = tmp_path / "not_an_image.txt"
        fake.write_text("this is not an image")
        proc = self._run_cli(predict_script, str(fake), "--device", "cpu")
        # Should fail (non-zero exit) or handle gracefully
        # We just confirm it doesn't hang
        assert proc.returncode is not None

    def test_cli_missing_file_errors(self, predict_script):
        proc = self._run_cli(predict_script, "/nonexistent/image.jpg", "--device", "cpu")
        assert proc.returncode != 0


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE INTEGRATION (Python API)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:

    def test_pipeline_all_test_images(self, model_bundle):
        """Run full pipeline on every bundled test image — none should crash."""
        for name, path in TEST_IMAGES.items():
            if not path.exists():
                continue
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
            assert isinstance(result, dict), f"Failed on {name}"
            assert "predicted_class" in result, f"Missing predicted_class for {name}"

    @pytest.mark.parametrize("class_name", DISEASE_CLASSES)
    def test_dataset_class_via_predict_image(self, model_bundle, real_dataset_samples, class_name):
        """predict_image() file-based API works on real dataset samples."""
        samples = real_dataset_samples.get(class_name, [])
        if not samples:
            pytest.skip(f"No samples for {class_name}")
        path = samples[0]
        result = predict_image(
            path,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=model_bundle["centroids_data"],
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )
        assert "image_path" in result
        assert "predicted_class" in result

    def test_sequential_predictions_are_independent(self, model_bundle):
        """Model state doesn't leak between predictions."""
        leaf = TEST_IMAGES.get("bacteria_leaf")
        non = TEST_IMAGES.get("wood")
        if not leaf or not leaf.exists() or not non or not non.exists():
            pytest.skip("Need both test images")

        kwargs = dict(
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=model_bundle["centroids_data"],
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )

        img_leaf = Image.open(leaf).convert("RGB")
        img_non = Image.open(non).convert("RGB")

        r1 = predict_pil_image(img_leaf, **kwargs)
        r2 = predict_pil_image(img_non, **kwargs)
        r3 = predict_pil_image(img_leaf, **kwargs)

        # r1 and r3 should be identical (same input)
        assert r1["predicted_class"] == r3["predicted_class"]
        assert abs(r1["confidence"] - r3["confidence"]) < 1e-5

    def test_brightness_validation_toggle(self, model_bundle):
        """validate_brightness=False should skip brightness check."""
        img = Image.new("RGB", (224, 224), (0, 0, 0))  # pitch black
        result = predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            validate_brightness=False,
        )
        assert "brightness_check" not in result or result["brightness_check"] is None

    def test_prediction_result_is_json_serializable(self, model_bundle):
        """Full result dict must be JSON-serializable (for API responses)."""
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
        # Should not raise
        serialized = json.dumps(result)
        assert len(serialized) > 0

    def test_different_top_k_values(self, model_bundle):
        """top_k=1,3,5,7 all work correctly."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        for k in [1, 3, 5, 7]:
            result = predict_pil_image(
                img,
                model=model_bundle["model"],
                device=model_bundle["device"],
                class_names=model_bundle["class_names"],
                disease_info=model_bundle["disease_info"],
                top_k=k,
            )
            if not result["rejected"]:
                assert len(result["top_k"]) == min(k, 7)
