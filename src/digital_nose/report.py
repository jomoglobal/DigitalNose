"""Scent report data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable


@dataclass
class ScentReport:
    """Structured summary for a single scent classification."""

    timestamp: datetime
    predicted_family: str
    confidence: float
    intensity_index: float
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable representation of the report."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "predicted_family": self.predicted_family,
            "confidence": round(self.confidence, 4),
            "intensity_index": round(self.intensity_index, 2),
            "raw_probabilities": {
                label: round(prob, 4) for label, prob in self.raw_probabilities.items()
            },
            "environment": {
                key: round(value, 2) for key, value in self.environment.items()
            },
        }

    @classmethod
    def from_prediction(
        cls,
        *,
        predicted_family: str,
        probabilities: Iterable[tuple[str, float]],
        intensity_index: float,
        environment: Dict[str, float],
    ) -> "ScentReport":
        """Create a report from prediction results."""

        prob_map = dict(probabilities)
        return cls(
            timestamp=datetime.utcnow(),
            predicted_family=predicted_family,
            confidence=float(prob_map[predicted_family]),
            intensity_index=float(intensity_index),
            raw_probabilities=prob_map,
            environment=environment,
        )


def intensity_from_reading(total_voc: float) -> float:
    """Compute a simple intensity score from total VOCs."""

    # Normalize by an arbitrary reference range to produce a 0-100 index.
    reference_max = 600.0
    scaled = total_voc / reference_max * 100
    return float(max(0.0, min(100.0, scaled)))
