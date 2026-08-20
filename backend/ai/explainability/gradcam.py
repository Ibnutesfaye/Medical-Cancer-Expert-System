"""
Grad-CAM — Gradient-weighted Class Activation Mapping
======================================================
Generates heatmaps showing which regions of a medical image
drove the model's cancer prediction.

Usage:
    from ai.explainability.gradcam import GradCAMGenerator
    gen = GradCAMGenerator(model)
    heatmap_b64 = gen.generate_b64(image_bytes, class_idx)

The heatmap is returned as a base64-encoded PNG overlay,
ready to embed directly in a frontend <img> tag.
"""

from __future__ import annotations

import io
import base64
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Same transform as inference (no Grayscale)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GradCAMGenerator:
    """Generates Grad-CAM heatmaps for a PyTorch CNN model."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Default target layer: last conv layer of ResNet
        if target_layer is None:
            # Works for ResNet18 and ResNet50
            if hasattr(model, 'layer4'):
                target_layer = model.layer4[-1]
            else:
                raise ValueError("Provide target_layer for non-ResNet models")

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_bytes: bytes,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap.

        Args:
            image_bytes: Raw image bytes
            class_idx:   Target class (None = predicted class)

        Returns:
            (heatmap, predicted_class_idx)
            heatmap: np.ndarray (224, 224) float32, values in [0, 1]
        """
        self.model.eval()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0)
        tensor.requires_grad_(True)

        output = self.model(tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    def generate_overlay(
        self,
        image_bytes: bytes,
        class_idx: Optional[int] = None,
        alpha: float = 0.5,
    ) -> tuple[bytes, int, list[dict]]:
        """
        Generate colorized Grad-CAM overlay blended with original image.

        Returns:
            (overlay_png_bytes, class_idx, key_regions)
        """
        import cv2

        cam, pred_class = self.generate(image_bytes, class_idx)

        # Original image resized
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)

        # Colorize heatmap (JET colormap: blue=low, red=high)
        heatmap_uint8 = (cam * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (img_array * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

        # Encode to PNG
        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        overlay_bytes = buf.getvalue()

        # Key regions (top activation areas)
        key_regions = self._extract_key_regions(cam)

        return overlay_bytes, pred_class, key_regions

    def generate_b64(
        self,
        image_bytes: bytes,
        class_idx: Optional[int] = None,
    ) -> tuple[str, int, list[dict]]:
        """Returns base64-encoded overlay PNG + key regions."""
        overlay_bytes, pred_class, regions = self.generate_overlay(image_bytes, class_idx)
        b64 = base64.b64encode(overlay_bytes).decode("utf-8")
        return b64, pred_class, regions

    def _extract_key_regions(self, cam: np.ndarray) -> list[dict]:
        """
        Identify top activation regions and assign anatomical labels.
        Returns list of {name, weight_pct} sorted by importance.
        """
        # Divide image into 3×3 grid and compute mean activation per cell
        h, w = cam.shape
        grid_h, grid_w = h // 3, w // 3

        region_names = [
            ["Top-Left",    "Top-Center",    "Top-Right"],
            ["Mid-Left",    "Center",        "Mid-Right"],
            ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
        ]
        regions = []
        for i in range(3):
            for j in range(3):
                cell = cam[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                regions.append({
                    "name":   region_names[i][j],
                    "weight": float(cell.mean()),
                })

        total = sum(r["weight"] for r in regions) or 1.0
        for r in regions:
            r["weight_pct"] = round(r["weight"] / total * 100, 1)
        regions.sort(key=lambda x: x["weight"], reverse=True)
        return [{"name": r["name"], "weight_pct": r["weight_pct"]}
                for r in regions[:4] if r["weight_pct"] > 1.0]
