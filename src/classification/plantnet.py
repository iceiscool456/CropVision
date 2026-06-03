"""
Pl@ntNet API client for in-the-wild species identification.

Sends a JPEG frame to the Pl@ntNet identify endpoint and returns ranked
species guesses.  Network/quota errors degrade gracefully (return an empty
list + a printed warning) so the live pipeline never crashes on a failed call.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import requests

from config.settings import (
    PLANTNET_API_KEY,
    PLANTNET_ENDPOINT,
    PLANTNET_TIMEOUT,
    PLANTNET_TOP_K,
)
from src.report import SpeciesGuess


class PlantNetClient:
    """Thin wrapper around the Pl@ntNet /v2/identify endpoint."""

    def __init__(self) -> None:
        self._api_key = PLANTNET_API_KEY
        if not self._api_key:
            print(
                "[PlantNet] WARNING: PLANTNET_API_KEY is not set. "
                "Add it to a .env file. Species ID will be unavailable."
            )

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def identify(self, bgr_image: np.ndarray) -> List[SpeciesGuess]:
        """Identify the plant species in a BGR image. Returns [] on failure."""
        if not self._api_key:
            return []

        ok, buffer = cv2.imencode(".jpg", bgr_image)
        if not ok:
            print("[PlantNet] Failed to encode frame as JPEG.")
            return []

        try:
            response = requests.post(
                PLANTNET_ENDPOINT,
                params={"api-key": self._api_key, "nb-results": PLANTNET_TOP_K},
                files={"images": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=PLANTNET_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"[PlantNet] Request failed: {exc}")
            return []

        if response.status_code == 429:
            print("[PlantNet] Daily quota exhausted (500/day on the free tier).")
            return []
        if response.status_code != 200:
            print(f"[PlantNet] API error {response.status_code}: {response.text[:200]}")
            return []

        return self._parse(response.json())

    @staticmethod
    def _parse(payload: dict) -> List[SpeciesGuess]:
        guesses: List[SpeciesGuess] = []
        for result in payload.get("results", []):
            species = result.get("species", {})
            common = species.get("commonNames") or [""]
            guesses.append(
                SpeciesGuess(
                    scientific_name=species.get("scientificNameWithoutAuthor", "Unknown"),
                    common_name=common[0],
                    score=float(result.get("score", 0.0)),
                )
            )
        remaining = payload.get("remainingIdentificationRequests")
        if remaining is not None:
            print(f"[PlantNet] {remaining} identifications remaining today.")
        return guesses
