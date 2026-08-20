"""
Integrated Gradients
====================
Computes feature importance by integrating gradients along a
straight-line path from a baseline (black image) to the input.

Returns a saliency map showing which pixels most influenced the prediction.
"""

from __future__ import annotations

import io
import base64
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def integrated_gradients(
    model: nn.Module,
    image_bytes: bytes,
    class_idx: int,
    steps: int = 50,
) -> np.ndarray:
    """
    Compute integrated gradients attribution map.

    Args:
        model:      PyTorch model
        image_bytes: Raw image bytes
        class_idx:  Target class index
        steps:      Number of interpolation steps (higher = more accurate)

    Returns:
        attribution map: np.ndarray (224, 224) float32, values in [-1, 1]
    """
    model.eval()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor  = _TRANSFORM(image).unsqueeze(0)           # (1, 3, 224, 224)
    baseline      = torch.zeros_like(input_tensor)            # black image

    # Interpolate between baseline and input
    alphas       = torch.linspace(0.0, 1.0, steps).view(-1, 1, 1, 1)
    interpolated = baseline + alphas * (input_tensor - baseline)  # (steps, 3, 224, 224)
    interpolated.requires_grad_(True)

    # Forward pass for all steps at once
    outputs = model(interpolated)
    target  = outputs[:, class_idx].sum()
    target.backward()

    gradients     = interpolated.grad.detach()           # (steps, 3, 224, 224)
    avg_gradients = gradients.mean(dim=0)                # (3, 224, 224)
    ig            = (input_tensor - baseline).squeeze() * avg_gradients  # (3, 224, 224)

    # Sum over channels → (224, 224)
    attribution = ig.abs().sum(dim=0).numpy()

    # Normalize to [0, 1]
    a_min, a_max = attribution.min(), attribution.max()
    if a_max > a_min:
        attribution = (attribution - a_min) / (a_max - a_min)

    return attribution


def integrated_gradients_b64(
    model: nn.Module,
    image_bytes: bytes,
    class_idx: int,
    steps: int = 50,
) -> str:
    """Returns base64 PNG of the attribution map overlaid on the image."""
    import cv2

    attr = integrated_gradients(model, image_bytes, class_idx, steps)

    image   = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_arr = np.array(image)

    attr_uint8 = (attr * 255).astype(np.uint8)
    attr_color = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_HOT)
    attr_rgb   = cv2.cvtColor(attr_color, cv2.COLOR_BGR2RGB)

    overlay = (img_arr * 0.5 + attr_rgb * 0.5).astype(np.uint8)
    pil_img = Image.fromarray(overlay)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
