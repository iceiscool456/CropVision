"""
Plant detector powered by YOLOv8.

Uses the ultralytics library to run the YOLO nano model.  Only detections
whose class ID is in PLANT_CLASS_IDS are forwarded downstream — everything
else (people, chairs, etc.) is silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from config.settings import (
    MODEL_CACHE_DIR,
    PLANT_CLASS_IDS,
    YOLO_CONFIDENCE,
    YOLO_MODEL_NAME,
)


@dataclass
class Detection:
    """A single detected plant region."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str


class PlantDetector:
    """Wraps YOLOv8 for plant-only detection."""

    def __init__(self) -> None:
        model_path = MODEL_CACHE_DIR / YOLO_MODEL_NAME
        self._model = YOLO(str(model_path) if model_path.exists() else YOLO_MODEL_NAME)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        detections: List[Detection] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in PLANT_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(box.conf[0]),
                        class_name=result.names[cls_id],
                    )
                )

        return detections
