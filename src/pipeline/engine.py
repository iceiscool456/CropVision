"""
Main processing pipeline: capture → detect → classify → visualize.

The engine ties all modules together in a single run-loop.  Classification
is intentionally throttled (runs every N frames) because transformer
inference is heavier than YOLO detection.

Supports two modes:
  - Live camera feed (default)
  - Single image file (--image path/to/file.jpg)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import CLASSIFY_EVERY_N_FRAMES, CLASSIFY_FULL_FRAME_FALLBACK
from src.camera.capture import Camera
from src.classification.classifier import Classification, PlantClassifier
from src.detection.detector import Detection, PlantDetector
from src.visualization.overlay import draw_detections, draw_fps, draw_fullframe_label


class PipelineEngine:
    """Real-time detect → classify → draw loop."""

    def __init__(self) -> None:
        print("[CropVision] Loading detection model …")
        self._detector = PlantDetector()
        print("[CropVision] Loading classification model …")
        self._classifier = PlantClassifier()
        print("[CropVision] Models loaded.")

    def run(self, image_path: Optional[str] = None) -> None:
        if image_path:
            self._run_on_image(Path(image_path))
        else:
            self._run_camera()

    def _run_on_image(self, path: Path) -> None:
        """Run the full pipeline on a single image file."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not decode image: {path}")

        print(f"[CropVision] Processing image: {path}")
        detections = self._detector.detect(frame)
        labels = self._classify_all(frame, detections)

        if detections:
            draw_detections(frame, detections, labels)
            print(f"[CropVision] Found {len(detections)} plant(s).")
        elif CLASSIFY_FULL_FRAME_FALLBACK:
            fallback = self._classifier.classify(frame)
            draw_fullframe_label(frame, fallback)
            print(f"[CropVision] No bbox — full-frame classification:")
            for c in fallback:
                print(f"  → {c.label}: {c.score:.1%}")

        print("[CropVision] Press any key to close.")
        cv2.imshow("CropVision", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _run_camera(self) -> None:
        """Block on the camera loop until the user presses 'q'."""
        frame_count = 0
        cached_labels: Dict[int, List[Classification]] = {}
        cached_fallback: List[Classification] = []
        prev_time = time.perf_counter()

        print("[CropVision] Starting camera …")
        with Camera() as cam:
            print("[CropVision] Pipeline running.  Press 'q' to quit.")
            while True:
                frame = cam.read()
                frame_count += 1

                # ── detect plants ────────────────────────────────────────
                detections: List[Detection] = self._detector.detect(frame)

                # ── classify species (throttled) ─────────────────────────
                if frame_count % CLASSIFY_EVERY_N_FRAMES == 0 or not cached_labels:
                    if detections:
                        cached_labels = self._classify_all(frame, detections)
                        cached_fallback = []
                    elif CLASSIFY_FULL_FRAME_FALLBACK:
                        cached_labels = {}
                        cached_fallback = self._classifier.classify(frame)
                    else:
                        cached_labels = {}
                        cached_fallback = []

                # ── draw overlays ────────────────────────────────────────
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-9)
                prev_time = now

                if detections:
                    draw_detections(frame, detections, cached_labels)
                elif cached_fallback:
                    draw_fullframe_label(frame, cached_fallback)
                draw_fps(frame, fps)

                cv2.imshow("CropVision", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()
        print("[CropVision] Shut down cleanly.")

    # ── private ──────────────────────────────────────────────────────────────

    def _classify_all(
        self,
        frame,
        detections: List[Detection],
    ) -> Dict[int, List[Classification]]:
        labels: Dict[int, List[Classification]] = {}
        for idx, det in enumerate(detections):
            crop = frame[det.y1 : det.y2, det.x1 : det.x2]
            if crop.size == 0:
                continue
            labels[idx] = self._classifier.classify(crop)
        return labels
