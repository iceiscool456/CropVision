"""
Shared result types produced by the pipeline.

Lives at the package root so the pipeline, the API client, and the
visualization layer can all import these without circular dependencies.

Two complementary signals per frame:
  * HealthReport  — local PlantVillage model: is a (crop) leaf healthy/diseased?
  * SpeciesGuess  — remote Pl@ntNet API: what species is this (in the wild)?
"""

from __future__ import annotations

from dataclasses import dataclass

# Health status constants
HEALTHY = "HEALTHY"
DISEASED = "DISEASED"
UNKNOWN = "UNKNOWN"


@dataclass
class HealthReport:
    """Health summary for one detected region (from the local disease model)."""

    status: str        # HEALTHY | DISEASED | UNKNOWN
    crop: str          # crop guess, or "" when unknown
    condition: str     # disease name, "healthy", or ""
    score: float

    def text(self) -> str:
        if self.status == UNKNOWN:
            # In leaf mode we keep a low-confidence guess around for feedback.
            if self.crop and self.score > 0:
                cond = self.condition or "unclear"
                return f"Unsure: {self.crop} - {cond} ({self.score:.0%})"
            return "Health: unknown"
        pct = f" {self.score:.0%}" if self.score else ""
        prefix = f"{self.crop}: " if self.crop else ""
        if self.status == HEALTHY:
            return f"{prefix}Healthy{pct}"
        return f"{prefix}Diseased ({self.condition}){pct}"


@dataclass
class SpeciesGuess:
    """One species candidate from Pl@ntNet."""

    scientific_name: str
    common_name: str
    score: float

    def text(self) -> str:
        name = self.common_name or self.scientific_name
        return f"{name} {self.score:.0%}"
