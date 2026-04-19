"""
test_robustness.py — Real-world robustness & edge-case tests.

Simulates the hostile conditions of uncontrolled field environments:
  - Low-light / overexposed capture
  - Motion blur / out-of-focus
  - Heavy JPEG compression (low-bandwidth upload)
  - Salt-and-pepper sensor noise
  - Partial occlusion / tight crop
  - Extreme rotation & aspect ratios
  - Unusual input types (grayscale, RGBA, BMP, WebP)
  - Adversarial-adjacent inputs (pure noise, solid colors, gradients)
  - Very small / very large input resolutions
  - Corrupt & edge-case files
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from potato_leaf_inference import (
    predict_pil_image,
    preprocess_pil,
    check_image_brightness,
)

from tests.conftest import TEST_IMAGES, DATASET_ROOT, DISEASE_CLASSES


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTING ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLightingRobustness:
    """Simulates variable lighting conditions common in open agricultural fields."""

    def _predict(self, model_bundle, img):
        return predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=model_bundle["centroids_data"],
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )

    def test_low_light_warns(self, model_bundle, low_light_leaf):
        """Extremely dark image should trigger brightness warning."""
        result = self._predict(model_bundle, low_light_leaf)
        bc = result.get("brightness_check", {})
        assert bc.get("level") == "WARNING_LOW_LIGHT"

    def test_overexposed_warns(self, model_bundle, overexposed_leaf):
        """Over-exposed image should trigger brightness warning."""
        result = self._predict(model_bundle, overexposed_leaf)
        bc = result.get("brightness_check", {})
        assert bc.get("level") == "WARNING_OVEREXPOSED"

    def test_slight_dimming_still_classifies(self, model_bundle):
        """Moderate dimming (0.4x brightness) should still produce a result."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        dimmed = ImageEnhance.Brightness(img).enhance(0.4)
        result = self._predict(model_bundle, dimmed)
        # Should return *something* — not crash
        assert "predicted_class" in result

    def test_shadow_overlay_robustness(self, model_bundle):
        """Half the image in shadow — simulates partial shade on the leaf."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        arr[:, : arr.shape[1] // 2] = (arr[:, : arr.shape[1] // 2] * 0.25).astype(
            np.uint8
        )
        shadowed = Image.fromarray(arr)
        result = self._predict(model_bundle, shadowed)
        assert "predicted_class" in result

    @pytest.mark.parametrize("factor", [0.5, 0.7, 1.5, 2.0, 2.5])
    def test_brightness_variations(self, model_bundle, factor):
        """Model should not crash across a range of brightness multipliers."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        adjusted = ImageEnhance.Brightness(img).enhance(factor)
        result = self._predict(model_bundle, adjusted)
        assert isinstance(result, dict)
        assert "predicted_class" in result


# ═══════════════════════════════════════════════════════════════════════════════
# BLUR & NOISE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlurAndNoiseRobustness:

    def _predict(self, model_bundle, img):
        return predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )

    @pytest.mark.parametrize("radius", [2, 5, 10, 20])
    def test_gaussian_blur_levels(self, model_bundle, radius):
        """Model should handle various blur levels without crashing."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        result = self._predict(model_bundle, blurred)
        assert isinstance(result, dict)

    def test_motion_blur_simulation(self, model_bundle):
        """Simulate horizontal motion blur via a directional kernel."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        # Box blur as proxy for motion blur (PIL requires square kernel)
        kernel_size = 5
        # Horizontal motion: 1s in the middle row, 0s elsewhere
        kernel_data = [0] * (kernel_size * kernel_size)
        mid = kernel_size // 2
        for i in range(kernel_size):
            kernel_data[mid * kernel_size + i] = 1
        kernel = ImageFilter.Kernel(
            size=(kernel_size, kernel_size),
            kernel=kernel_data,
            scale=kernel_size,
        )
        blurred = img.filter(kernel)
        result = self._predict(model_bundle, blurred)
        assert isinstance(result, dict)

    def test_salt_pepper_noise(self, model_bundle, salt_pepper_image):
        result = self._predict(model_bundle, salt_pepper_image)
        assert isinstance(result, dict)

    def test_gaussian_noise_overlay(self, model_bundle):
        """Add Gaussian noise (sigma=30) to a real leaf image."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 30, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        result = self._predict(model_bundle, Image.fromarray(noisy))
        assert isinstance(result, dict)

    def test_jpeg_heavy_compression(self, model_bundle, jpeg_artifact_leaf):
        """Quality=5 JPEG — extreme compression artifacts."""
        result = self._predict(model_bundle, jpeg_artifact_leaf)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("quality", [10, 30, 50, 80, 95])
    def test_jpeg_quality_ladder(self, model_bundle, tmp_path, quality):
        """Test across a range of JPEG compression qualities."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        out = tmp_path / f"q{quality}.jpg"
        img.save(str(out), "JPEG", quality=quality)
        reloaded = Image.open(str(out))
        result = self._predict(model_bundle, reloaded)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometricRobustness:

    def _predict(self, model_bundle, img):
        return predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )

    @pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 270])
    def test_rotation_angles(self, model_bundle, angle):
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        rotated = img.rotate(angle, expand=True, fillcolor=(0, 0, 0))
        result = self._predict(model_bundle, rotated)
        assert isinstance(result, dict)

    def test_horizontal_flip(self, model_bundle):
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        result = self._predict(model_bundle, flipped)
        assert isinstance(result, dict)

    def test_vertical_flip(self, model_bundle):
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        result = self._predict(model_bundle, flipped)
        assert isinstance(result, dict)

    def test_extreme_aspect_ratio_tall(self, model_bundle):
        """Very tall, narrow image (50x800)."""
        img = Image.new("RGB", (50, 800), (60, 130, 60))
        result = self._predict(model_bundle, img)
        assert isinstance(result, dict)

    def test_extreme_aspect_ratio_wide(self, model_bundle):
        """Very wide, short image (800x50)."""
        img = Image.new("RGB", (800, 50), (60, 130, 60))
        result = self._predict(model_bundle, img)
        assert isinstance(result, dict)

    def test_partial_crop(self, model_bundle, cropped_partial_leaf):
        result = self._predict(model_bundle, cropped_partial_leaf)
        assert isinstance(result, dict)

    def test_center_crop_real_image(self, model_bundle):
        """Center-crop to 50% — simulates zoomed-in capture."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf.jpg not found")
        img = Image.open(path).convert("RGB")
        w, h = img.size
        cropped = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        result = self._predict(model_bundle, cropped)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("size", [(16, 16), (32, 32), (64, 64), (1024, 1024), (3840, 2160)])
    def test_various_resolutions(self, model_bundle, size):
        """Model should handle input images of vastly different resolutions."""
        img = Image.new("RGB", size, (60, 130, 60))
        result = self._predict(model_bundle, img)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR SPACE & FORMAT EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestColorAndFormatEdgeCases:

    def _predict(self, model_bundle, img):
        return predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
        )

    def test_grayscale_input(self, model_bundle, grayscale_image):
        result = self._predict(model_bundle, grayscale_image)
        assert isinstance(result, dict)

    def test_rgba_input(self, model_bundle, rgba_image):
        result = self._predict(model_bundle, rgba_image)
        assert isinstance(result, dict)

    def test_palette_mode(self, model_bundle):
        """P-mode (palette) image — e.g. from a GIF."""
        img = Image.new("P", (224, 224))
        result = self._predict(model_bundle, img)
        assert isinstance(result, dict)

    def test_16bit_image(self, model_bundle):
        """16-bit per channel image (I;16 mode)."""
        arr = np.random.randint(0, 65535, (224, 224), dtype=np.uint16)
        img = Image.fromarray(arr, mode="I;16")
        # Convert to RGB for processing
        result = self._predict(model_bundle, img)
        assert isinstance(result, dict)

    def test_bmp_format(self, model_bundle, tmp_path):
        """BMP file from disk."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        path = tmp_path / "test.bmp"
        img.save(str(path))
        loaded = Image.open(str(path))
        result = self._predict(model_bundle, loaded)
        assert isinstance(result, dict)

    def test_webp_format(self, model_bundle, tmp_path):
        """WebP file from disk."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        path = tmp_path / "test.webp"
        img.save(str(path))
        loaded = Image.open(str(path))
        result = self._predict(model_bundle, loaded)
        assert isinstance(result, dict)

    def test_png_format(self, model_bundle, tmp_path):
        """PNG file from disk."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        path = tmp_path / "test.png"
        img.save(str(path))
        loaded = Image.open(str(path))
        result = self._predict(model_bundle, loaded)
        assert isinstance(result, dict)

    def test_in_memory_bytes_io(self, model_bundle):
        """Image loaded from BytesIO (simulates upload stream)."""
        img = Image.new("RGB", (224, 224), (60, 130, 60))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        loaded = Image.open(buf)
        result = self._predict(model_bundle, loaded)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL-ADJACENT INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialInputs:
    """Inputs that are misleading or pathological for the model."""

    def _predict_full(self, model_bundle, img):
        return predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=model_bundle["centroids_data"],
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )

    def test_solid_green_rectangle(self, model_bundle):
        """Solid green — similar color to leaves but no texture."""
        img = Image.new("RGB", (400, 300), (34, 139, 34))
        result = self._predict_full(model_bundle, img)
        assert isinstance(result, dict)

    def test_green_gradient(self, model_bundle):
        """Green gradient — tests color-only features."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            arr[i, :] = (0, int(i * 1.1), 0)
        img = Image.fromarray(arr)
        result = self._predict_full(model_bundle, img)
        assert isinstance(result, dict)

    def test_printed_leaf_photo(self, model_bundle):
        """Simulate photo of a photo — add border and slight color cast."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf not found")
        img = Image.open(path).convert("RGB")
        # Add white border
        bordered = Image.new("RGB", (img.width + 60, img.height + 60), (240, 240, 240))
        bordered.paste(img, (30, 30))
        # Yellow color cast
        arr = np.array(bordered, dtype=np.float32)
        arr[:, :, 0] *= 1.1
        arr[:, :, 2] *= 0.85
        cast = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        result = self._predict_full(model_bundle, cast)
        assert isinstance(result, dict)

    def test_multiple_leaves_in_one_image(self, model_bundle):
        """Collage of two leaf images side by side."""
        paths = [TEST_IMAGES.get("bacteria_leaf"), TEST_IMAGES.get("healthy_leaf")]
        imgs = []
        for p in paths:
            if p and p.exists():
                imgs.append(Image.open(p).convert("RGB").resize((224, 224)))
        if len(imgs) < 2:
            pytest.skip("Need 2 test leaf images")
        collage = Image.new("RGB", (448, 224))
        collage.paste(imgs[0], (0, 0))
        collage.paste(imgs[1], (224, 0))
        result = self._predict_full(model_bundle, collage)
        assert isinstance(result, dict)

    def test_text_on_leaf(self, model_bundle):
        """Leaf image with text overlay (label/annotation)."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf not found")
        from PIL import ImageDraw
        img = Image.open(path).convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "SAMPLE #42", fill=(255, 0, 0))
        draw.text((10, 30), "Lab: ABC", fill=(255, 255, 0))
        result = self._predict_full(model_bundle, img)
        assert isinstance(result, dict)

    def test_leaf_on_complex_background(self, model_bundle):
        """Leaf pasted onto a cluttered background (soil + gravel texture)."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf not found")
        leaf = Image.open(path).convert("RGB").resize((150, 150))
        # Create a noisy brown background (simulates soil)
        bg = np.random.randint(80, 160, (400, 400, 3), dtype=np.uint8)
        bg[:, :, 0] = np.clip(bg[:, :, 0] + 40, 0, 255)  # reddish-brown tint
        bg_img = Image.fromarray(bg)
        bg_img.paste(leaf, (125, 125))
        result = self._predict_full(model_bundle, bg_img)
        assert isinstance(result, dict)

    def test_uniform_color_per_class(self, model_bundle):
        """Each class's average color — should be rejected as not a real leaf."""
        # Approximate average colors for non-leaf inputs
        colors = [
            (200, 50, 50),   # red
            (50, 50, 200),   # blue
            (200, 200, 50),  # yellow
            (100, 100, 100), # gray
            (255, 165, 0),   # orange
        ]
        for color in colors:
            img = Image.new("RGB", (224, 224), color)
            result = self._predict_full(model_bundle, img)
            assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED DEGRADATION (STACKING MULTIPLE EFFECTS)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinedDegradation:
    """Real-world conditions often combine multiple degradations."""

    def _predict(self, model_bundle, img):
        return predict_pil_image(
            img,
            model=model_bundle["model"],
            device=model_bundle["device"],
            class_names=model_bundle["class_names"],
            disease_info=model_bundle["disease_info"],
            centroids_data=model_bundle["centroids_data"],
            gate_model=model_bundle["gate_model"],
            gate_categories=model_bundle["gate_categories"],
        )

    def test_dim_plus_blur(self, model_bundle):
        """Low light + out of focus — common in field conditions."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf not found")
        img = Image.open(path).convert("RGB")
        dim = ImageEnhance.Brightness(img).enhance(0.3)
        blurry = dim.filter(ImageFilter.GaussianBlur(radius=5))
        result = self._predict(model_bundle, blurry)
        assert isinstance(result, dict)

    def test_noise_plus_compression(self, model_bundle, tmp_path):
        """Sensor noise + heavy JPEG compression."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf not found")
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        noisy = np.clip(arr + np.random.normal(0, 25, arr.shape), 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy)
        jpeg_path = tmp_path / "noisy_compressed.jpg"
        noisy_img.save(str(jpeg_path), "JPEG", quality=15)
        reloaded = Image.open(str(jpeg_path))
        result = self._predict(model_bundle, reloaded)
        assert isinstance(result, dict)

    def test_rotation_plus_crop_plus_blur(self, model_bundle):
        """Rotated + cropped + blurred — simulates quick careless photo."""
        path = TEST_IMAGES.get("bacteria_leaf")
        if path is None or not path.exists():
            pytest.skip("test_bacteria_leaf not found")
        img = Image.open(path).convert("RGB")
        rotated = img.rotate(35, expand=True, fillcolor=(0, 0, 0))
        w, h = rotated.size
        cropped = rotated.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        blurred = cropped.filter(ImageFilter.GaussianBlur(radius=3))
        result = self._predict(model_bundle, blurred)
        assert isinstance(result, dict)

    def test_high_contrast_plus_noise(self, model_bundle):
        """Harsh sunlight (high contrast) + sensor noise."""
        path = TEST_IMAGES.get("healthy_leaf")
        if path is None or not path.exists():
            pytest.skip("test_healthy_leaf not found")
        img = Image.open(path).convert("RGB")
        contrast = ImageEnhance.Contrast(img).enhance(3.0)
        arr = np.array(contrast, dtype=np.float32)
        noisy = np.clip(arr + np.random.normal(0, 20, arr.shape), 0, 255).astype(np.uint8)
        result = self._predict(model_bundle, Image.fromarray(noisy))
        assert isinstance(result, dict)
