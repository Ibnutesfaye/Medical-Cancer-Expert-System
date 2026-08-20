"""
Label Classes Loader
Reads label_classes.json and exposes idx_to_class mapping.
Used by FHE server and benchmark service.
"""

import json
from pathlib import Path
from functools import lru_cache

LABELS_PATH = Path(__file__).parent / "label_classes.json"


@lru_cache(maxsize=1)
def get_idx_to_class() -> dict[int, str]:
    with open(LABELS_PATH) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["idx_to_class"].items()}


@lru_cache(maxsize=1)
def get_class_to_idx() -> dict[str, int]:
    with open(LABELS_PATH) as f:
        data = json.load(f)
    return data["class_to_idx"]


@lru_cache(maxsize=1)
def get_cancer_labels() -> dict[str, str]:
    with open(LABELS_PATH) as f:
        data = json.load(f)
    return data["cancer_labels"]
