"""Machine learning pipeline for the Digital Nose sample app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sensors import ENVIRONMENT_FEATURES, VOC_FEATURES

FEATURE_COLUMNS = VOC_FEATURES + ENVIRONMENT_FEATURES


@dataclass
class ModelArtifacts:
    """Container for the trained model and metadata."""

    pipeline: Pipeline
    feature_columns: Iterable[str]
    classes_: Iterable[str]


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the sklearn pipeline used for training and inference."""

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURE_COLUMNS)]
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    return pipeline


def train_model(df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42) -> Tuple[ModelArtifacts, Dict[str, str]]:
    """Train the classifier and return the model plus evaluation metrics."""

    X = df[FEATURE_COLUMNS]
    y = df["scent_family"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    artifacts = ModelArtifacts(
        pipeline=pipeline, feature_columns=FEATURE_COLUMNS, classes_=pipeline.classes_
    )
    return artifacts, {"classification_report": report}


def predict(artifacts: ModelArtifacts, sample: Dict[str, float]) -> Tuple[str, np.ndarray]:
    """Predict the scent family for a single reading."""

    X = pd.DataFrame([sample])[FEATURE_COLUMNS]
    probabilities = artifacts.pipeline.predict_proba(X)[0]
    prediction_idx = np.argmax(probabilities)
    return artifacts.pipeline.classes_[prediction_idx], probabilities
