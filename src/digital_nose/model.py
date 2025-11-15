"""Lightweight scent classification helpers for the Digital Nose sample app."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .sensors import ENVIRONMENT_FEATURES, VOC_FEATURES

FEATURE_COLUMNS = VOC_FEATURES + ENVIRONMENT_FEATURES


@dataclass
class ModelArtifacts:
    """Container for the trained model and metadata."""

    class_means: Dict[str, Dict[str, float]]
    feature_columns: Iterable[str]
    classes_: List[str]


def _compute_class_means(data: Sequence[Dict[str, float | str]]) -> Dict[str, Dict[str, float]]:
    class_totals: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for row in data:
        label = str(row["scent_family"])
        counts[label] = counts.get(label, 0) + 1
        totals = class_totals.setdefault(
            label, {feature: 0.0 for feature in FEATURE_COLUMNS}
        )
        for feature in FEATURE_COLUMNS:
            totals[feature] += float(row[feature])

    class_means: Dict[str, Dict[str, float]] = {}
    for label, totals in class_totals.items():
        count = max(counts[label], 1)
        class_means[label] = {
            feature: totals[feature] / count for feature in FEATURE_COLUMNS
        }
    return class_means


def _predict_from_means(
    class_means: Dict[str, Dict[str, float]], sample: Dict[str, float]
) -> Tuple[str, Dict[str, float]]:
    """Predict a scent family using distance to class means."""

    distances: List[Tuple[str, float]] = []
    for label, centroid in class_means.items():
        distance = math.sqrt(
            sum(
                (float(sample[feature]) - centroid[feature]) ** 2
                for feature in FEATURE_COLUMNS
            )
        )
        distances.append((label, distance))

    # Convert distances into normalized scores. Closer means higher probability.
    scores: Dict[str, float] = {}
    for label, distance in distances:
        scores[label] = 1.0 / (distance + 1e-6)
    total = sum(scores.values()) or 1.0
    probabilities = {label: score / total for label, score in scores.items()}

    predicted_label = max(probabilities.items(), key=lambda item: item[1])[0]
    return predicted_label, probabilities


def train_model(
    data: Sequence[Dict[str, float | str]],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ModelArtifacts, Dict[str, object]]:
    """Train the centroid-based classifier and return model artifacts plus metrics."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset = [dict(row) for row in data]
    if not dataset:
        raise ValueError("Dataset must contain rows to train the model.")

    rng = random.Random(random_state)
    rng.shuffle(dataset)

    if len(dataset) < 2:
        train_data = dataset
        test_data = dataset
    else:
        holdout_count = max(1, int(round(len(dataset) * test_size)))
        if holdout_count >= len(dataset):
            holdout_count = len(dataset) - 1
        test_data = dataset[:holdout_count]
        train_data = dataset[holdout_count:]
        if not train_data:
            train_data = dataset

    class_means = _compute_class_means(train_data)
    artifacts = ModelArtifacts(
        class_means=class_means,
        feature_columns=FEATURE_COLUMNS,
        classes_=sorted(class_means.keys()),
    )

    evaluation: Dict[str, Dict[str, int]] = {
        label: {"correct": 0, "total": 0} for label in artifacts.classes_
    }
    total_samples = len(test_data)
    correct_predictions = 0

    for row in test_data:
        actual_label = str(row["scent_family"])
        sample = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
        predicted_label, _ = _predict_from_means(class_means, sample)
        evaluation.setdefault(actual_label, {"correct": 0, "total": 0})
        evaluation[actual_label]["total"] += 1
        if predicted_label == actual_label:
            evaluation[actual_label]["correct"] += 1
            correct_predictions += 1

    overall_accuracy = (
        correct_predictions / total_samples if total_samples else 0.0
    )

    per_class_accuracy = {
        label: (
            evaluation[label]["correct"] / evaluation[label]["total"]
            if evaluation[label]["total"]
            else None
        )
        for label in sorted(evaluation.keys())
    }

    metrics: Dict[str, object] = {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "samples_evaluated": total_samples,
    }

    return artifacts, metrics


def predict(
    artifacts: ModelArtifacts, sample: Dict[str, float]
) -> Tuple[str, Dict[str, float]]:
    """Predict the scent family for a single reading."""

    filtered_sample = {
        feature: float(sample[feature]) for feature in artifacts.feature_columns
    }
    return _predict_from_means(artifacts.class_means, filtered_sample)
