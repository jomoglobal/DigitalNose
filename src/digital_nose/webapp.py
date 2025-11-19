"""Flask-based web interface for the Digital Nose sample app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from .dataset import load_dataset
from .model import predict, train_model
from .report import ScentReport, intensity_from_reading
from .sensors import (
    ENVIRONMENT_FEATURES,
    VOC_FEATURES,
    DEFAULT_PROFILES,
    SensorSimulator,
    ScentProfile,
)

# Initialize Flask app
app = Flask(__name__)

# Global state
simulator = SensorSimulator()
dataset: list[Dict[str, Any]] = []
artifacts: Any = None
metrics: Dict[str, Any] = {}
profile_map: Dict[str, ScentProfile] = {
    profile.name: profile for profile in DEFAULT_PROFILES
}


def initialize_model() -> None:
    """Load dataset and train the model."""
    global dataset, artifacts, metrics
    dataset = load_dataset()
    artifacts, metrics = train_model(dataset)


# Routes
@app.route("/")
def index() -> str:
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/api/init", methods=["GET"])
def api_init() -> Any:
    """Initialize the application and return configuration."""
    if artifacts is None:
        initialize_model()

    return jsonify({
        "profiles": list(profile_map.keys()),
        "voc_features": VOC_FEATURES,
        "environment_features": ENVIRONMENT_FEATURES,
        "metrics": {
            "overall_accuracy": metrics.get("overall_accuracy", 0.0),
            "per_class_accuracy": metrics.get("per_class_accuracy", {}),
        },
        "classes": artifacts.classes_ if artifacts else [],
    })


@app.route("/api/capture", methods=["POST"])
def api_capture() -> Any:
    """Capture a sample and return the classification results."""
    data = request.get_json()
    profile_name = data.get("profile")

    if profile_name not in profile_map:
        return jsonify({"error": "Unknown profile"}), 400

    profile = profile_map[profile_name]
    reading = simulator.capture(profile, n_samples=1)[0]

    total_voc = sum(reading[feature] for feature in VOC_FEATURES)
    environment = {feature: reading[feature] for feature in ENVIRONMENT_FEATURES}

    predicted_family, probabilities = predict(artifacts, reading)
    report = ScentReport.from_prediction(
        predicted_family=predicted_family,
        probabilities=probabilities.items(),
        intensity_index=intensity_from_reading(total_voc),
        environment=environment,
    )

    # Prepare response
    return jsonify({
        "success": True,
        "reading": {feature: reading[feature] for feature in VOC_FEATURES},
        "report": {
            "predicted_family": report.predicted_family,
            "confidence": report.confidence,
            "intensity_index": report.intensity_index,
            "environment": report.environment,
            "raw_probabilities": report.raw_probabilities,
            "timestamp": report.timestamp.isoformat(),
        },
    })


def main(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Start the Flask development server."""
    print(f"Starting Digital Nose Web UI...")
    print(f"Open your browser to: http://{host}:{port}")
    initialize_model()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main(debug=True)
