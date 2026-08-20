"""
Feature Extractor for Concrete ML FHE Pipeline
================================================
Runs ResNet18 locally (on the server or patient device) to extract
a 512-dimensional feature vector from a medical image.

This is Step 1 of the hybrid FHE architecture:
  Patient device  →  ResNet18 feature extraction (plaintext)
                  →  Quantize to int8
                  →  FHE encrypt
                  →  Send to server

The server NEVER sees the raw image — only the encrypted feature vector.
"""

from __future__ import annotations

import io
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Transform — must match training transform (no Grayscale)
# ---------------------------------------------------------------------------
FEATURE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ResNet18FeatureExtractor:
    """
    Extracts 512-dim feature vectors from medical images using
    a frozen pretrained ResNet18 backbone (no fine-tuning needed).

    In the FHE workflow this runs on the CLIENT (patient device).
    In demo/server mode it runs on the server for convenience.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        # Load ResNet18, strip the final FC layer
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.extractor = nn.Sequential(*list(base.children())[:-1])
        self.extractor.eval()
        self.extractor.to(self.device)

    @torch.no_grad()
    def extract(self, image_bytes: bytes) -> np.ndarray:
        """
        Extract 512-dim feature vector from raw image bytes.

        Args:
            image_bytes: Raw image file bytes (JPEG/PNG/etc.)

        Returns:
            np.ndarray of shape (512,), dtype float32
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = FEATURE_TRANSFORM(image).unsqueeze(0).to(self.device)
        features = self.extractor(tensor)          # (1, 512, 1, 1)
        return features.squeeze().cpu().numpy()    # (512,)

    def extract_batch(self, images: list[bytes]) -> np.ndarray:
        """
        Extract features from a batch of images.

        Returns:
            np.ndarray of shape (N, 512)
        """
        tensors = []
        for img_bytes in images:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(FEATURE_TRANSFORM(image))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.extractor(batch).squeeze(-1).squeeze(-1)
        return features.cpu().numpy()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_extractor: ResNet18FeatureExtractor | None = None


def get_feature_extractor() -> ResNet18FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ResNet18FeatureExtractor()
    return _extractor
