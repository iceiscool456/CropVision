"""
Headless analysis service shared by the web API (and reusable elsewhere).

Wraps the three models — YOLO detector, local disease model, Pl@ntNet species
API — and turns a single BGR frame into a plain JSON-serializable dict. No
OpenCV windows or camera loop here, so it works equally well for an uploaded
photo or a live browser frame.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from config.settings import (
    CLASSIFY_FULL_FRAME_FALLBACK,
    HEALTH_CONFIDENCE,
)
from src.classification.plantnet import PlantNetClient
from src.detection.detector import Detection, PlantDetector
from src.health.analyzer import HealthAnalyzer
from src.health.plantid import PlantIdClient
from src.report import UNKNOWN, DeepHealth, HealthReport, SpeciesGuess


def _health_for(health: HealthAnalyzer, region: np.ndarray) -> HealthReport:
    """Gate the disease model: only trust crops above HEALTH_CONFIDENCE."""
    if region.size == 0:
        return HealthReport(UNKNOWN, "", "", 0.0)
    results = health.analyze(region)
    top = results[0] if results else None
    if top is not None and top.score >= HEALTH_CONFIDENCE:
        return HealthReport(top.status, top.crop, top.condition, top.score)
    return HealthReport(UNKNOWN, "", "", 0.0)


def _health_dict(report: HealthReport, box: Optional[tuple]) -> dict:
    """Serialize a HealthReport for the quick local health read."""
    return {
        "status": report.status,
        "crop": report.crop,
        "condition": report.condition,
        "score": round(report.score, 4),
        "text": report.text(),
        "box": list(box) if box else None,
    }


def _species_dict(guess: SpeciesGuess) -> dict:
    return {
        "scientific_name": guess.scientific_name,
        "common_name": guess.common_name,
        "score": round(guess.score, 4),
        "text": guess.text(),
    }


class AnalysisService:
    """Loads the models once and analyzes individual frames into JSON dicts."""

    def __init__(self) -> None:
        print("[LeafLens] Loading detection model …")
        self.detector = PlantDetector()
        print("[LeafLens] Loading health/disease model …")
        self.health = HealthAnalyzer()
        print("[LeafLens] Connecting Pl@ntNet species API …")
        self.plantnet = PlantNetClient()
        self.plantid = PlantIdClient()
        print("[LeafLens] Ready.")

    @property
    def species_enabled(self) -> bool:
        return self.plantnet.enabled

    @property
    def deep_enabled(self) -> bool:
        return self.plantid.enabled

    def deep_credits(self) -> Optional[dict]:
        return self.plantid.usage()

    def deep_diagnose(self, bgr: np.ndarray) -> Optional[dict]:
        """Accurate remote disease diagnosis (Plant.id). None if disabled/failed."""
        assessment: Optional[DeepHealth] = self.plantid.assess(bgr)
        if assessment is None:
            return None
        return {
            "is_healthy": assessment.is_healthy,
            "is_healthy_probability": round(assessment.is_healthy_probability, 4),
            "suggestions": [
                {
                    "name": s.name,
                    "probability": round(s.probability, 4),
                    "description": s.description,
                    "treatment": s.treatment,
                }
                for s in assessment.suggestions
            ],
        }

    @property
    def supported_crops(self) -> List[str]:
        return sorted(self.health.supported_crops)

    def analyze(
        self,
        bgr: np.ndarray,
        identify_species: bool = False,
    ) -> dict:
        """Analyze one BGR frame. Returns a JSON-serializable result dict."""
        h, w = bgr.shape[:2]
        out: dict = {
            "image_size": [w, h],
            "detections": [],
            "health": [],
            "species": [],
            "species_enabled": self.plantnet.enabled,
        }

        detections: List[Detection] = self.detector.detect(bgr)
        if detections:
            for d in detections:
                box = (d.x1, d.y1, d.x2, d.y2)
                report = _health_for(self.health, bgr[d.y1:d.y2, d.x1:d.x2])
                out["detections"].append(
                    {
                        "box": list(box),
                        "confidence": round(d.confidence, 4),
                        "class_name": d.class_name,
                    }
                )
                out["health"].append(_health_dict(report, box=box))
        elif CLASSIFY_FULL_FRAME_FALLBACK:
            report = _health_for(self.health, bgr)
            out["health"].append(_health_dict(report, box=None))

        if identify_species and self.plantnet.enabled:
            out["species"] = [_species_dict(g) for g in self.plantnet.identify(bgr)]

        return out
