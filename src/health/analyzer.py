"""
Plant health analyzer.

Wraps a MobileNetV2 classifier trained on PlantVillage (38 classes across 14
crops).  Labels are human-readable, in three shapes:

  * ``Tomato with Late Blight``      → crop=Tomato,  condition=Late Blight (diseased)
  * ``Healthy Tomato Plant``         → crop=Tomato,  condition=healthy
  * ``Apple Scab`` / ``Tomato Mosaic Virus`` → crop implied, condition is the rest

From each prediction we derive the crop species, the condition, and a coarse
health status (HEALTHY vs DISEASED).

Because the model only knows these 14 crops, callers should gate on the
returned ``score``: a high score means the plant really is one of the supported
crops; a low score means it is probably something else, so health is reported
as UNKNOWN (species identification is handled separately by Pl@ntNet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from config.settings import HEALTH_MODEL, HEALTH_TOP_K
from src.report import DISEASED, HEALTHY

# Crop display names the model knows.  Order matters for substring matching:
# longer / more specific names must come first (e.g. "Corn (Maize)" before any
# shorter overlap).
KNOWN_CROPS: Tuple[str, ...] = (
    "Corn (Maize)",
    "Bell Pepper",
    "Apple",
    "Blueberry",
    "Cherry",
    "Grape",
    "Orange",
    "Peach",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
)


@dataclass
class HealthResult:
    crop: str          # e.g. "Tomato"
    condition: str     # e.g. "Late Blight" or "healthy"
    status: str        # HEALTHY | DISEASED
    score: float
    raw_label: str


def _parse_label(label: str) -> Tuple[str, str, str]:
    """Return (crop, condition, status) for one of the 38 model labels."""
    is_healthy = label.strip().lower().startswith("healthy")

    crop = next((c for c in KNOWN_CROPS if c in label), "Unknown")
    crop_display = crop.replace(" (Maize)", "")  # "Corn (Maize)" -> "Corn"

    if is_healthy:
        return crop_display, "healthy", HEALTHY

    # Diseased: strip the crop name and the connector word "with" to leave the
    # condition, then collapse whitespace.
    remainder = label.replace(crop, "").replace("with", "", 1)
    condition = " ".join(remainder.split()) or label
    return crop_display, condition, DISEASED


class HealthAnalyzer:
    """Runs the PlantVillage MobileNetV2 model and parses crop/condition/status."""

    def __init__(self) -> None:
        self._pipe = pipeline(
            "image-classification", model=HEALTH_MODEL, top_k=HEALTH_TOP_K
        )

    @property
    def supported_crops(self) -> set:
        """The set of crop species this model can recognize."""
        return {c.replace(" (Maize)", "") for c in KNOWN_CROPS}

    def analyze(self, bgr_crop: np.ndarray) -> List[HealthResult]:
        """Return top-K health results for a BGR image crop."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        raw = self._pipe(pil_img)

        results: List[HealthResult] = []
        for pred in raw:
            crop, condition, status = _parse_label(pred["label"])
            results.append(
                HealthResult(
                    crop=crop,
                    condition=condition,
                    status=status,
                    score=float(pred["score"]),
                    raw_label=pred["label"],
                )
            )
        return results
