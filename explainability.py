"""
Explainability module — Grad-CAM and Score-CAM for the Hybrid CNN-Transformer.

Provides visual explanations of model predictions by generating class-activation
heatmaps overlaid on the input image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from potato_leaf_inference import (
    HybridCNNTransformer,
    get_val_transforms,
    IMAGE_SIZE,
)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for HybridCNNTransformer.

    Hooks into the last convolutional layer of the EfficientNetB0 backbone
    and computes the weighted activation map for a target class.
    """

    def __init__(self, model: HybridCNNTransformer):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Register hooks on the backbone's final conv block output
        self._fwd_hook = model.backbone.conv_head.register_forward_hook(self._save_activation)
        self._bwd_hook = model.backbone.conv_head.register_full_backward_hook(self._save_gradient)

    # -- hook callbacks -----------------------------------------------------
    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # -- public API ---------------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Return a 2-D heatmap (H, W) in [0, 1] for *target_class*.

        If *target_class* is None the predicted class is used.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward for the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        # Weighted combination of activation channels
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalise to [0, 1] and resize to input size
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        denom = cam.max() - cam.min()
        if denom > 0:
            cam = cam / denom
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR
            )
        ) / 255.0
        return cam

    def release(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Score-CAM  (gradient-free)
# ---------------------------------------------------------------------------

class ScoreCAM:
    """Score-weighted Class Activation Mapping (gradient-free).

    Uses forward-pass confidence changes of each activation channel as
    weights instead of gradients — more stable and noise-free.
    """

    def __init__(self, model: HybridCNNTransformer, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self._activations: torch.Tensor | None = None

        self._fwd_hook = model.backbone.conv_head.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Return a 2-D heatmap (H, W) in [0, 1]."""
        self.model.eval()
        device = input_tensor.device

        # Baseline forward — get activations + predicted class
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        activations = self._activations  # [1, C, h, w]
        C = activations.shape[1]
        H, W = IMAGE_SIZE

        # Up-sample each channel to input resolution and use as mask
        upsampled = F.interpolate(
            activations, size=(H, W), mode="bilinear", align_corners=False,
        )[0]  # [C, H, W]

        # Normalise each channel independently to [0, 1]
        for c in range(C):
            ch = upsampled[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 0:
                upsampled[c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                upsampled[c] = torch.zeros_like(ch)

        # Score each mask by the model's confidence on target_class
        scores = torch.zeros(C, device=device)

        for start in range(0, C, self.batch_size):
            end = min(start + self.batch_size, C)
            masks = upsampled[start:end].unsqueeze(1)  # [bs, 1, H, W]
            masked = input_tensor * masks  # broadcast
            out = self.model(masked)
            probs = F.softmax(out, dim=1)
            scores[start:end] = probs[:, target_class]

        # Weighted combination
        weights = scores.view(-1, 1, 1)  # [C, 1, 1]
        cam = (weights * upsampled).sum(dim=0)  # [H, W]
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        denom = cam.max() - cam.min()
        if denom > 0:
            cam = cam / denom
        return cam

    def release(self):
        self._fwd_hook.remove()


# ---------------------------------------------------------------------------
# Overlay utility
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a [0,1] heatmap onto *original* using matplotlib colormapping."""
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]  # drop alpha channel
    colored = (colored * 255).astype(np.uint8)
    colored_pil = Image.fromarray(colored).resize(original.size, Image.Resampling.BILINEAR)

    blended = Image.blend(original.convert("RGB").resize(original.size), colored_pil, alpha)
    return blended


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_explanation(
    img: Image.Image,
    model: HybridCNNTransformer,
    device: torch.device,
    class_names: list[str],
    method: str = "gradcam",
    target_class: int | None = None,
    alpha: float = 0.5,
) -> dict:
    """High-level API: produce a heatmap + overlay for an image.

    Parameters
    ----------
    method : "gradcam" | "scorecam"
    target_class : class index; None → use predicted class.

    Returns
    -------
    dict with keys: heatmap (np.ndarray), overlay (PIL.Image),
    predicted_class, target_class, confidence.
    """
    transform = get_val_transforms()
    input_tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

    # Get prediction first
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    if target_class is None:
        target_class = pred_idx

    if method == "scorecam":
        explainer = ScoreCAM(model)
    else:
        explainer = GradCAM(model)

    try:
        heatmap = explainer.generate(input_tensor, target_class=target_class)
    finally:
        explainer.release()

    overlay = overlay_heatmap(img.convert("RGB").resize(IMAGE_SIZE), heatmap, alpha=alpha)

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "target_class": class_names[target_class],
        "target_index": target_class,
        "confidence": float(probs[pred_idx]),
    }
