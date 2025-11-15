"""Sensor simulation utilities for the Digital Nose sample app."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

VOC_FEATURES = [
    "acetone_ppb",
    "ethanol_ppb",
    "toluene_ppb",
    "ammonia_ppb",
    "hydrogen_sulfide_ppb",
    "terpene_ppb",
]

ENVIRONMENT_FEATURES = ["temperature_c", "humidity_pct"]


@dataclass
class SensorConfig:
    """Configuration for the simulated sensor."""

    baseline_noise: float = 0.05
    drift_rate: float = 0.01
    sample_rate_hz: int = 1


@dataclass
class ScentProfile:
    """Idealized feature distribution for a scent family."""

    name: str
    mean_vector: Dict[str, float]
    variance_vector: Dict[str, float]


DEFAULT_PROFILES: List[ScentProfile] = [
    ScentProfile(
        name="citrus",
        mean_vector={
            "acetone_ppb": 120.0,
            "ethanol_ppb": 80.0,
            "toluene_ppb": 5.0,
            "ammonia_ppb": 3.0,
            "hydrogen_sulfide_ppb": 2.0,
            "terpene_ppb": 150.0,
            "temperature_c": 23.0,
            "humidity_pct": 40.0,
        },
        variance_vector={feature: 0.1 for feature in VOC_FEATURES + ENVIRONMENT_FEATURES},
    ),
    ScentProfile(
        name="herbal",
        mean_vector={
            "acetone_ppb": 35.0,
            "ethanol_ppb": 60.0,
            "toluene_ppb": 15.0,
            "ammonia_ppb": 10.0,
            "hydrogen_sulfide_ppb": 4.0,
            "terpene_ppb": 90.0,
            "temperature_c": 22.5,
            "humidity_pct": 50.0,
        },
        variance_vector={feature: 0.15 for feature in VOC_FEATURES + ENVIRONMENT_FEATURES},
    ),
    ScentProfile(
        name="woody",
        mean_vector={
            "acetone_ppb": 45.0,
            "ethanol_ppb": 35.0,
            "toluene_ppb": 30.0,
            "ammonia_ppb": 6.0,
            "hydrogen_sulfide_ppb": 3.5,
            "terpene_ppb": 200.0,
            "temperature_c": 21.0,
            "humidity_pct": 45.0,
        },
        variance_vector={feature: 0.12 for feature in VOC_FEATURES + ENVIRONMENT_FEATURES},
    ),
    ScentProfile(
        name="sweet",
        mean_vector={
            "acetone_ppb": 15.0,
            "ethanol_ppb": 95.0,
            "toluene_ppb": 8.0,
            "ammonia_ppb": 4.0,
            "hydrogen_sulfide_ppb": 2.5,
            "terpene_ppb": 170.0,
            "temperature_c": 22.0,
            "humidity_pct": 48.0,
        },
        variance_vector={feature: 0.08 for feature in VOC_FEATURES + ENVIRONMENT_FEATURES},
    ),
]


class SensorSimulator:
    """Simulate a multi-channel VOC sensor with gradual drift."""

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()
        self._rng = random.Random()
        self._tick = 0

    def capture(self, profile: ScentProfile, *, n_samples: int = 1) -> pd.DataFrame:
        """Simulate `n_samples` sensor captures for the provided profile."""

        rows = []
        for _ in range(n_samples):
            rows.append(self._simulate_single(profile))
            self._tick += 1
        return pd.DataFrame(rows)

    def _simulate_single(self, profile: ScentProfile) -> Dict[str, float]:
        reading: Dict[str, float] = {}
        drift = self.config.drift_rate * math.sin(self._tick / 25.0)
        for feature in VOC_FEATURES + ENVIRONMENT_FEATURES:
            baseline = profile.mean_vector[feature]
            variance = profile.variance_vector.get(feature, 0.1)
            noise = (self._rng.random() - 0.5) * variance * 2
            reading[feature] = max(0.0, baseline * (1 + noise + drift * self.config.baseline_noise))
        reading["scent_family"] = profile.name
        return reading


def sample_dataset(
    *, profiles: Iterable[ScentProfile] | None = None,
    samples_per_profile: int = 120,
    simulator: SensorSimulator | None = None,
) -> pd.DataFrame:
    """Generate a labeled dataset of simulated sensor readings."""

    profiles = list(profiles or DEFAULT_PROFILES)
    simulator = simulator or SensorSimulator()
    frames = [simulator.capture(profile, n_samples=samples_per_profile) for profile in profiles]
    return pd.concat(frames, ignore_index=True)
