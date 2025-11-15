"""Command-line entry point for the Digital Nose sample app."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.table import Table

from .dataset import ensure_dataset, load_dataset
from .model import ModelArtifacts, predict, train_model
from .report import ScentReport, intensity_from_reading
from .sensors import ENVIRONMENT_FEATURES, VOC_FEATURES, SensorSimulator, ScentProfile

console = Console()


def _print_header(title: str) -> None:
    console.rule(f"[bold green]{title}")


def train_and_evaluate(dataset_path: Path) -> tuple[ModelArtifacts, dict[str, str]]:
    df = load_dataset(dataset_path)
    artifacts, metrics = train_model(df)
    console.print("[bold cyan]Model trained.[/]")
    console.print(metrics["classification_report"])
    return artifacts, metrics


def simulate_live_reading(artifacts: ModelArtifacts, profile: ScentProfile) -> ScentReport:
    simulator = SensorSimulator()
    reading = simulator.capture(profile, n_samples=1).iloc[0].to_dict()
    total_voc = sum(reading[feature] for feature in VOC_FEATURES)
    env = {feature: reading[feature] for feature in ENVIRONMENT_FEATURES}
    predicted_family, probabilities = predict(artifacts, reading)
    report = ScentReport.from_prediction(
        predicted_family=predicted_family,
        probabilities=zip(artifacts.classes_, probabilities),
        intensity_index=intensity_from_reading(total_voc),
        environment=env,
    )
    return report


def display_report(report: ScentReport) -> None:
    table = Table(title="Scent Report")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in report.as_dict().items():
        if isinstance(value, dict):
            table.add_row(key, json.dumps(value, indent=2))
        else:
            table.add_row(key, str(value))
    console.print(table)


def main(args: Iterable[str] | None = None) -> None:
    dataset_path = ensure_dataset()
    _print_header("Digital Nose Sample App")
    artifacts, _ = train_and_evaluate(dataset_path)

    console.print("\n[bold green]Simulating live scent capture...[/]")
    sample_profile = random.choice(list(_available_profiles()))
    report = simulate_live_reading(artifacts, sample_profile)
    display_report(report)
    console.print("\nSaved dataset at:", dataset_path)
    console.print("Run `digital-nose-app` again to simulate another reading.")


def _available_profiles() -> Iterable[ScentProfile]:
    from .sensors import DEFAULT_PROFILES

    return DEFAULT_PROFILES


if __name__ == "__main__":  # pragma: no cover
    main()
