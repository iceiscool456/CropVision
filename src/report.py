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
        """Quick local health label (crop + best-guess disease)."""
        if self.status == UNKNOWN:
            return "Health: unknown"
        pct = f" {self.score:.0%}" if self.score else ""
        prefix = f"{self.crop}: " if self.crop else ""
        if self.status == HEALTHY:
            return f"{prefix}Healthy{pct}"
        # "likely" signals the specific disease name is a best guess.
        return f"{prefix}Diseased (likely {self.condition}){pct}"


@dataclass
class SpeciesGuess:
    """One species candidate from Pl@ntNet."""

    scientific_name: str
    common_name: str
    score: float

    def text(self) -> str:
        name = self.common_name or self.scientific_name
        return f"{name} {self.score:.0%}"


@dataclass
class DiseaseSuggestion:
    """One disease/pest candidate from the Plant.id health API."""

    name: str
    probability: float
    description: str
    treatment: str  # flattened, human-readable treatment guidance

    def text(self) -> str:
        return f"{self.name} {self.probability:.0%}"


@dataclass
class DeepHealth:
    """Accurate (remote) health assessment from Plant.id."""

    is_healthy: bool
    is_healthy_probability: float
    suggestions: list  # List[DiseaseSuggestion]
