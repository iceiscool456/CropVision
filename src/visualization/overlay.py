"""
Drawing utilities: health-colored boxes + an on-demand species banner.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from config.settings import (
    BOX_COLOR,
    BOX_THICKNESS,
    LABEL_BG_COLOR,
    LABEL_COLOR,
    LABEL_FONT_SCALE,
    STATUS_COLORS,
)
from src.detection.detector import Detection
from src.report import HealthReport, SpeciesGuess


def draw_health(
    frame: np.ndarray,
    detections: List[Detection],
    reports: List[HealthReport],
) -> np.ndarray:
    """Draw a health-colored box + label for each detection."""
    for det, rep in zip(detections, reports):
        color = STATUS_COLORS.get(rep.status, BOX_COLOR)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, BOX_THICKNESS)
        _draw_label(frame, rep.text(), det.x1, det.y1 - 8, color)
    return frame


def draw_health_fullframe(frame: np.ndarray, report: HealthReport) -> np.ndarray:
    """Health label anchored bottom-left when there is no bounding box."""
    color = STATUS_COLORS.get(report.status, BOX_COLOR)
    _draw_label(frame, report.text(), 10, frame.shape[0] - 20, color)
    return frame


def draw_species_banner(
    frame: np.ndarray,
    guesses: List[SpeciesGuess],
    identifying: bool = False,
) -> np.ndarray:
    """Top-left species panel populated by Pl@ntNet (on-demand)."""
    if identifying:
        _draw_label(frame, "Identifying species …", 10, 60, BOX_COLOR)
        return frame
    if not guesses:
        _draw_label(frame, "Press 'i' to identify species", 10, 60, BOX_COLOR)
        return frame

    _draw_label(frame, "Species (Pl@ntNet):", 10, 60, BOX_COLOR)
    y = 60
    for guess in guesses:
        y += 26
        _draw_label(frame, f"  {guess.text()}", 10, y, BOX_COLOR)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    _draw_label(frame, f"FPS: {fps:.1f}", 10, 30, BOX_COLOR)
    return frame


# ── private helpers ──────────────────────────────────────────────────────────

def _ascii(text: str) -> str:
    """OpenCV's Hershey fonts are ASCII-only; map/strip non-ASCII to avoid '???'."""
    return (
        text.replace("×", "x")
        .replace("–", "-")
        .replace("—", "-")
        .encode("ascii", "ignore")
        .decode()
    )


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    accent: tuple = LABEL_BG_COLOR,
) -> None:
    text = _ascii(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, LABEL_FONT_SCALE, 1)
    x = max(x, 0)
    y = max(y, th + 4)
    cv2.rectangle(
        frame, (x, y - th - 4), (x + tw + 6, y + baseline), LABEL_BG_COLOR, -1
    )
    cv2.rectangle(frame, (x, y - th - 4), (x + 3, y + baseline), accent, -1)
    cv2.putText(
        frame, text, (x + 6, y - 2), font, LABEL_FONT_SCALE, LABEL_COLOR, 1
    )
