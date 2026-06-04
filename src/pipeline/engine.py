"""
Main processing pipeline: capture → detect → (health locally) + (species on demand).

Two signals are produced:
  * Health  — the local PlantVillage model runs continuously and colors each
    detected region green/red/gray (healthy / diseased / unknown).
  * Species — the Pl@ntNet API is called ON DEMAND (press 'i' in live mode,
    automatic in --image mode) to keep within the free 500/day quota and avoid
    per-frame network latency.

Modes:
  - Live camera feed (default):  'i' identify · 's' snapshot · 'q' quit
  - Single image file (--image path/to/file.jpg)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config.settings import (
    CLASSIFY_EVERY_N_FRAMES,
    CLASSIFY_FULL_FRAME_FALLBACK,
    HEALTH_CONFIDENCE,
)
from src.camera.capture import Camera
from src.classification.plantnet import PlantNetClient
from src.detection.detector import Detection, PlantDetector
from src.health.analyzer import HealthAnalyzer
from src.report import UNKNOWN, HealthReport, SpeciesGuess
from src.visualization.overlay import (
    draw_fps,
    draw_health,
    draw_health_fullframe,
    draw_species_banner,
)


class PipelineEngine:
    """Real-time detect → diagnose loop with on-demand species ID."""

    def __init__(self) -> None:
        print("[LeafLens] Loading detection model …")
        self._detector = PlantDetector()
        print("[LeafLens] Loading health/disease model …")
        self._health = HealthAnalyzer()
        print("[LeafLens] Connecting Pl@ntNet species API …")
        self._plantnet = PlantNetClient()
        print("[LeafLens] Ready.")

    def run(self, image_path: Optional[str] = None) -> None:
        if image_path:
            self._run_on_image(Path(image_path))
        else:
            self._run_camera()

    # ── image mode ─────────────────────────────────────────────────────────
    def _run_on_image(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not decode image: {path}")

        print(f"[LeafLens] Processing image: {path}")
        detections = self._detector.detect(frame)

        if detections:
            reports = [self._health_for(frame[d.y1:d.y2, d.x1:d.x2]) for d in detections]
            draw_health(frame, detections, reports)
            for r in reports:
                print(f"  • health: {r.text()}")
        elif CLASSIFY_FULL_FRAME_FALLBACK:
            report = self._health_for(frame)
            draw_health_fullframe(frame, report)
            print(f"  • health: {report.text()}")

        # Species ID runs automatically for a single image.
        print("[LeafLens] Identifying species via Pl@ntNet …")
        guesses = self._plantnet.identify(frame)
        draw_species_banner(frame, guesses)
        for g in guesses:
            print(f"  • species: {g.text()}  ({g.scientific_name})")

        print("[LeafLens] Press any key to close.")
        cv2.imshow("LeafLens", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── camera mode ────────────────────────────────────────────────────────
    def _run_camera(self) -> None:
        frame_count = 0
        cached_reports: List[HealthReport] = []
        cached_full: Optional[HealthReport] = None
        species: List[SpeciesGuess] = []
        prev_time = time.perf_counter()

        print("[LeafLens] Starting camera …")
        with Camera() as cam:
            print("[LeafLens] Running.  'i' species · 's' snapshot · 'q' quit.")
            while True:
                frame = cam.read()
                frame_count += 1
                analyze = (frame_count - 1) % CLASSIFY_EVERY_N_FRAMES == 0

                detections: List[Detection] = self._detector.detect(frame)
                if analyze:
                    if detections:
                        cached_reports = [
                            self._health_for(frame[d.y1:d.y2, d.x1:d.x2])
                            for d in detections
                        ]
                        cached_full = None
                    elif CLASSIFY_FULL_FRAME_FALLBACK:
                        cached_reports = []
                        cached_full = self._health_for(frame)
                    else:
                        cached_reports = []
                        cached_full = None

                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-9)
                prev_time = now

                if detections:
                    draw_health(frame, detections, cached_reports)
                elif cached_full is not None:
                    draw_health_fullframe(frame, cached_full)
                draw_species_banner(frame, species)
                draw_fps(frame, fps)

                cv2.imshow("LeafLens", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    self._save_snapshot(frame)
                if key == ord("i"):
                    draw_species_banner(frame, [], identifying=True)
                    cv2.imshow("LeafLens", frame)
                    cv2.waitKey(1)
                    print("[LeafLens] Identifying species via Pl@ntNet …")
                    species = self._plantnet.identify(frame)
                    for g in species:
                        print(f"  • {g.text()}  ({g.scientific_name})")

        cv2.destroyAllWindows()
        print("[LeafLens] Shut down cleanly.")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _health_for(self, region: np.ndarray) -> HealthReport:
        """Run the local disease model and collapse it to a HealthReport."""
        if region.size == 0:
            return HealthReport(UNKNOWN, "", "", 0.0)
        results = self._health.analyze(region)
        top = results[0] if results else None
        if top is not None and top.score >= HEALTH_CONFIDENCE:
            return HealthReport(
                status=top.status,
                crop=top.crop,
                condition=top.condition,
                score=top.score,
            )
        return HealthReport(UNKNOWN, "", "", 0.0)

    @staticmethod
    def _save_snapshot(frame: np.ndarray) -> None:
        captures = Path("captures")
        captures.mkdir(exist_ok=True)
        out = captures / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"[LeafLens] Saved snapshot → {out}")
