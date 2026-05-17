"""
Drawing utilities for bounding boxes and text labels on frames.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from config.settings import (
    BOX_COLOR,
    BOX_THICKNESS,
    LABEL_BG_COLOR,
    LABEL_COLOR,
    LABEL_FONT_SCALE,
)
from src.classification.classifier import Classification
from src.detection.detector import Detection


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    labels: dict[int, List[Classification]] | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes and species labels onto *frame* (mutated in place).

    Parameters
    ----------
    frame : np.ndarray
        The BGR video frame.
    detections : list[Detection]
        Plant bounding boxes from the detector.
    labels : dict[int, list[Classification]] | None
        Mapping from detection index → classification results.  If None or
        missing for a given index, only the YOLO class name is shown.
    """
    for idx, det in enumerate(detections):
        cv2.rectangle(
            frame, (det.x1, det.y1), (det.x2, det.y2), BOX_COLOR, BOX_THICKNESS
        )

        if labels and idx in labels and labels[idx]:
            top = labels[idx][0]
            text = f"{_pretty(top.label)} {top.score:.0%}"
        else:
            text = f"{det.class_name} {det.confidence:.0%}"

        _draw_label(frame, text, det.x1, det.y1 - 8)

    return frame


def draw_fullframe_label(
    frame: np.ndarray,
    classifications: List[Classification],
) -> np.ndarray:
    """When no bbox was found, show species predictions anchored to the bottom."""
    h = frame.shape[0]
    y = h - 20
    for cls in reversed(classifications):
        text = f"{_pretty(cls.label)} {cls.score:.0%}"
        _draw_label(frame, text, 10, y)
        y -= 28
    if classifications:
        _draw_label(frame, "Full-frame ID (no bbox)", 10, y)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    _draw_label(frame, f"FPS: {fps:.1f}", 10, 30)
    return frame


# ── private helpers ──────────────────────────────────────────────────────────

def _pretty(label: str) -> str:
    """Clean up a model label for display.

    Handles both ImageNet-style ('pot, flowerpot') and botanical names
    ('Euphorbia_aggregata_A_Berger' → 'Euphorbia Aggregata').
    """
    name = label.split(",")[0].strip().replace("_", " ")
    parts = name.split()
    # Keep only the genus + species (first two words), drop author citations
    if len(parts) >= 2 and parts[0][0].isupper():
        name = " ".join(parts[:2])
    return name.title()


def _draw_label(frame: np.ndarray, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, LABEL_FONT_SCALE, 1)
    y = max(y, th + 4)
    cv2.rectangle(
        frame, (x, y - th - 4), (x + tw + 4, y + baseline), LABEL_BG_COLOR, -1
    )
    cv2.putText(frame, text, (x + 2, y - 2), font, LABEL_FONT_SCALE, LABEL_COLOR, 1)
