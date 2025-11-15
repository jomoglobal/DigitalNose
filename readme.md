# Digital Nose Sample Application

This repository contains a self-contained sample implementation of the **Digital Nose** concept. It simulates VOC sensor readings, trains a lightweight classifier, and produces human-readable scent reports from new captures. A graphical dashboard lets you explore simulated captures interactively.

## Features

- Synthetic data generator that produces repeatable “scent fingerprints” for four scent families (citrus, herbal, woody, sweet).
- Dataset utilities that build and persist a CSV dataset of simulated readings.
- Lightweight centroid classifier that provides confidence scores without heavy dependencies.
- Command-line app that trains the model, simulates a new capture, and prints a standardized scent report.
- Tkinter dashboard with interactive controls, probability breakdowns, and live VOC fingerprint visualization.
- Unit test to validate that the training pipeline runs end-to-end.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -e .
   ```

2. **Run the console sample app**

   ```bash
   digital-nose-app
   ```

   The first run generates a dataset at `data/sample_scent_readings.csv`, trains the classifier, and prints a scent report for a simulated capture.

3. **Launch the graphical dashboard**

   ```bash
   digital-nose-gui
   ```

   Use the "Capture Sample" button to simulate captures for different scent families and view the resulting predictions, sensor fingerprints, and environment readings.

4. **Run tests**

   ```bash
   pytest
   ```

## Project Structure

```
src/digital_nose/
├── app.py           # CLI entry point
├── dataset.py       # Dataset generation and loading helpers
├── gui.py           # Tkinter dashboard for interactive exploration
├── model.py         # Lightweight centroid classifier helpers
├── report.py        # Scent report dataclass and utilities
└── sensors.py       # Sensor simulator and scent profiles
```

`tests/` contains unit tests, and `data/` stores generated datasets.

## Extending the Sample

- Add new `ScentProfile` entries in `sensors.py` to introduce more scent families.
- Swap `RandomForestClassifier` with your preferred model inside `model.py`.
- Build an API or dashboard by reusing the existing dataset/model utilities.

Have fun experimenting with digital scent analytics!
