"""Dataset utilities for the Digital Nose sample app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .sensors import DEFAULT_PROFILES, SensorSimulator, ScentProfile, sample_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "sample_scent_readings.csv"


def ensure_dataset(
    *,
    path: Path = DEFAULT_DATASET_PATH,
    profiles: Iterable[ScentProfile] | None = None,
    samples_per_profile: int = 120,
    force: bool = False,
) -> Path:
    """Make sure a dataset exists at the given path, generating if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or force:
        df = sample_dataset(
            profiles=profiles or DEFAULT_PROFILES,
            samples_per_profile=samples_per_profile,
            simulator=SensorSimulator(),
        )
        df.to_csv(path, index=False)
    return path


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the dataset, generating it first if necessary."""

    ensure_dataset(path=path)
    return pd.read_csv(path)
