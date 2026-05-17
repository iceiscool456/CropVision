"""
Plant species classifier using a HuggingFace Vision Transformer.

Given a cropped BGR image of a detected plant, this module returns the
top-K predicted species labels with confidence scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from config.settings import (
    CLASSIFIER_CONFIDENCE,
    CLASSIFIER_MODEL,
    CLASSIFIER_TOP_K,
)


@dataclass
class Classification:
    label: str
    score: float


class PlantClassifier:
    """Runs a HuggingFace image-classification pipeline on plant crops."""

    def __init__(self) -> None:
        self._pipe = pipeline(
            "image-classification",
            model=CLASSIFIER_MODEL,
            top_k=CLASSIFIER_TOP_K,
        )

    def classify(self, bgr_crop: np.ndarray) -> List[Classification]:
        """Classify a BGR numpy crop and return filtered predictions."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        raw = self._pipe(pil_img)
        return [
            Classification(label=p["label"], score=p["score"])
            for p in raw
            if p["score"] >= CLASSIFIER_CONFIDENCE
        ]
