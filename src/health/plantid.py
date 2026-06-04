"""
Plant.id (Kindwise) v3 health-assessment client.

Submits a photo to the plant.health API and returns an accurate, in-the-wild
disease diagnosis — specific diseases/pests (Fungi, Bacteria, Viruses, …) with
probabilities and treatment guidance. This is the high-accuracy counterpart to
the local PlantVillage model, used on demand ("Deep diagnosis").

Degrades gracefully: with no API key, or on any network/quota error, it returns
None and prints a warning so the rest of the app keeps working.
"""

from __future__ import annotations

import base64
from typing import List, Optional

import cv2
import numpy as np
import requests

from config.settings import (
    PLANTID_API_KEY,
    PLANTID_ENDPOINT,
    PLANTID_TIMEOUT,
    PLANTID_TOP_K,
    PLANTID_USAGE_ENDPOINT,
)
from src.report import DeepHealth, DiseaseSuggestion


class PlantIdClient:
    """Thin wrapper around the Plant.id /v3/health_assessment endpoint."""

    def __init__(self) -> None:
        self._api_key = PLANTID_API_KEY

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def usage(self) -> Optional[dict]:
        """Fetch remaining credits (free endpoint — costs nothing). None on fail."""
        if not self._api_key:
            return None
        try:
            response = requests.get(
                PLANTID_USAGE_ENDPOINT,
                headers={"Api-Key": self._api_key},
                timeout=PLANTID_TIMEOUT,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        data = response.json()
        return {
            "used": data.get("used", {}).get("total"),
            "total": data.get("credit_limits", {}).get("total"),
            "remaining": data.get("remaining", {}).get("total"),
        }

    def assess(self, bgr_image: np.ndarray) -> Optional[DeepHealth]:
        """Diagnose disease in a BGR image. Returns None on failure/disabled."""
        if not self._api_key:
            return None

        ok, buffer = cv2.imencode(".jpg", bgr_image)
        if not ok:
            print("[Plant.id] Failed to encode frame as JPEG.")
            return None
        data_url = "data:image/jpeg;base64," + base64.b64encode(
            buffer.tobytes()
        ).decode()

        try:
            response = requests.post(
                PLANTID_ENDPOINT,
                params={"details": "description,treatment", "language": "en"},
                headers={"Api-Key": self._api_key, "Content-Type": "application/json"},
                json={"images": [data_url]},
                timeout=PLANTID_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"[Plant.id] Request failed: {exc}")
            return None

        if response.status_code in (401, 403):
            print("[Plant.id] Auth failed — check PLANTID_API_KEY.")
            return None
        if response.status_code in (402, 429):
            print("[Plant.id] Out of credits (100 free, then paid).")
            return None
        if response.status_code not in (200, 201):
            print(f"[Plant.id] API error {response.status_code}: {response.text[:200]}")
            return None

        return self._parse(response.json())

    @staticmethod
    def _flatten_treatment(details: dict) -> str:
        treatment = (details or {}).get("treatment") or {}
        parts: List[str] = []
        for key in ("prevention", "biological", "chemical"):
            items = treatment.get(key)
            if items:
                parts.append(f"{key.capitalize()}: " + "; ".join(items[:2]))
        return "  •  ".join(parts)

    @classmethod
    def _parse(cls, payload: dict) -> Optional[DeepHealth]:
        result = payload.get("result", {})
        is_healthy = result.get("is_healthy", {}) or {}
        disease = result.get("disease", {}) or {}

        suggestions: List[DiseaseSuggestion] = []
        for s in disease.get("suggestions", [])[:PLANTID_TOP_K]:
            details = s.get("details", {}) or {}
            suggestions.append(
                DiseaseSuggestion(
                    name=s.get("name", "Unknown"),
                    probability=float(s.get("probability", 0.0)),
                    description=details.get("description") or "",
                    treatment=cls._flatten_treatment(details),
                )
            )

        return DeepHealth(
            is_healthy=bool(is_healthy.get("binary", False)),
            is_healthy_probability=float(is_healthy.get("probability", 0.0)),
            suggestions=suggestions,
        )
