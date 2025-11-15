"""Dataset utilities for the Digital Nose sample app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

from .sensors import (
    DEFAULT_PROFILES,
    ENVIRONMENT_FEATURES,
    SensorSimulator,
    ScentProfile,
    VOC_FEATURES,
    sample_dataset,
)

FIELD_ORDER = VOC_FEATURES + ENVIRONMENT_FEATURES + ["scent_family"]

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
        rows = sample_dataset(
            profiles=profiles or DEFAULT_PROFILES,
            samples_per_profile=samples_per_profile,
            simulator=SensorSimulator(),
        )
        with path.open("w", newline="") as handle:
            writer = _csv_writer(handle)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in FIELD_ORDER})
    return path


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> List[Dict[str, float | str]]:
    """Load the dataset, generating it first if necessary."""

    ensure_dataset(path=path)
    dataset: List[Dict[str, float | str]] = []
    with path.open(newline="") as handle:
        reader = _csv_reader(handle)
        for row in reader:
            entry: Dict[str, float | str] = {}
            for field in VOC_FEATURES + ENVIRONMENT_FEATURES:
                entry[field] = float(row[field])
            entry["scent_family"] = row["scent_family"]
            dataset.append(entry)
    return dataset


def _csv_writer(handle):
    import csv

    return csv.DictWriter(handle, fieldnames=FIELD_ORDER)


def _csv_reader(handle):
    import csv

    return csv.DictReader(handle)
