"""Tkinter-based graphical interface for the Digital Nose sample app."""

from __future__ import annotations

import itertools
import tkinter as tk
from tkinter import ttk
from typing import Dict

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


class DigitalNoseApp(tk.Tk):
    """Simple dashboard for exploring the Digital Nose simulation."""

    CHART_HEIGHT = 260
    CHART_WIDTH = 420
    CHART_MARGIN = 24

    def __init__(self) -> None:
        super().__init__()
        self.title("Digital Nose Visual Simulator")
        self.geometry("960x640")
        self.minsize(900, 600)

        self.simulator = SensorSimulator()
        self.dataset = load_dataset()
        self.artifacts, self.metrics = train_model(self.dataset)

        self.profile_map: Dict[str, ScentProfile] = {
            profile.name: profile for profile in DEFAULT_PROFILES
        }
        self.profile_var = tk.StringVar(value=DEFAULT_PROFILES[0].name)

        self.report: ScentReport | None = None

        self._build_layout()
        self._update_metrics_panel()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.style = ttk.Style()
        self.style.configure("Metrics.TLabel", font=("Helvetica", 12, "bold"))
        self.style.configure("Result.TLabel", font=("Helvetica", 13))

        container = ttk.Frame(self, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            container,
            text="Digital Nose Simulator",
            font=("Helvetica", 20, "bold"),
        )
        header.pack(anchor=tk.W)

        description = ttk.Label(
            container,
            text=(
                "Generate virtual scent captures, classify them, and explore "
                "sensor responses visually."
            ),
            wraplength=700,
        )
        description.pack(anchor=tk.W, pady=(0, 12))

        controls_frame = ttk.Frame(container)
        controls_frame.pack(fill=tk.X, pady=(0, 16))

        ttk.Label(controls_frame, text="Choose scent family:").pack(side=tk.LEFT)
        self.profile_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.profile_var,
            values=list(self.profile_map.keys()),
            state="readonly",
            width=20,
        )
        self.profile_combo.pack(side=tk.LEFT, padx=(6, 12))

        self.capture_button = ttk.Button(
            controls_frame,
            text="Capture Sample",
            command=self.capture_sample,
        )
        self.capture_button.pack(side=tk.LEFT)

        self.status_label = ttk.Label(controls_frame, text="Ready.")
        self.status_label.pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(container)
        body.pack(fill=tk.BOTH, expand=True)

        chart_card = ttk.LabelFrame(body, text="VOC fingerprint")
        chart_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.chart_canvas = tk.Canvas(
            chart_card,
            width=self.CHART_WIDTH,
            height=self.CHART_HEIGHT,
            bg="white",
            highlightthickness=0,
        )
        self.chart_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        right_panel = ttk.Frame(body)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.metrics_panel = ttk.LabelFrame(right_panel, text="Model performance")
        self.metrics_panel.pack(fill=tk.X, padx=(0, 12))

        result_card = ttk.LabelFrame(right_panel, text="Latest classification")
        result_card.pack(fill=tk.BOTH, expand=True, padx=(0, 12), pady=(12, 0))

        self.result_label = ttk.Label(result_card, style="Result.TLabel")
        self.result_label.pack(anchor=tk.W, padx=12, pady=(12, 4))

        intensity_frame = ttk.Frame(result_card)
        intensity_frame.pack(fill=tk.X, padx=12)
        ttk.Label(intensity_frame, text="Intensity index:").pack(side=tk.LEFT)
        self.intensity_var = tk.DoubleVar(value=0.0)
        self.intensity_bar = ttk.Progressbar(
            intensity_frame,
            variable=self.intensity_var,
            maximum=100,
            length=180,
        )
        self.intensity_bar.pack(side=tk.LEFT, padx=8)
        self.intensity_label = ttk.Label(intensity_frame, text="0.0")
        self.intensity_label.pack(side=tk.LEFT)

        ttk.Separator(result_card).pack(fill=tk.X, padx=12, pady=8)

        env_frame = ttk.Frame(result_card)
        env_frame.pack(fill=tk.X, padx=12)
        ttk.Label(env_frame, text="Environment:").pack(anchor=tk.W)
        self.environment_vars = {
            feature: tk.StringVar(value="–") for feature in ENVIRONMENT_FEATURES
        }
        for feature in ENVIRONMENT_FEATURES:
            row = ttk.Frame(env_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{feature.replace('_', ' ').title()}:", width=18).pack(
                side=tk.LEFT
            )
            ttk.Label(row, textvariable=self.environment_vars[feature]).pack(side=tk.LEFT)

        ttk.Separator(result_card).pack(fill=tk.X, padx=12, pady=8)

        prob_frame = ttk.Frame(result_card)
        prob_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        ttk.Label(prob_frame, text="Class probabilities:").pack(anchor=tk.W)
        self.probability_tree = ttk.Treeview(
            prob_frame,
            columns=("label", "score"),
            show="headings",
            height=5,
        )
        self.probability_tree.heading("label", text="Scent family")
        self.probability_tree.heading("score", text="Confidence")
        self.probability_tree.column("label", anchor=tk.W, width=140)
        self.probability_tree.column("score", anchor=tk.CENTER, width=100)
        self.probability_tree.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def capture_sample(self) -> None:
        profile_name = self.profile_var.get()
        profile = self.profile_map.get(profile_name)
        if profile is None:
            self.status_label.config(text="Unknown profile selected.")
            return

        reading = self.simulator.capture(profile, n_samples=1)[0]
        total_voc = sum(reading[feature] for feature in VOC_FEATURES)
        environment = {feature: reading[feature] for feature in ENVIRONMENT_FEATURES}

        predicted_family, probabilities = predict(self.artifacts, reading)
        report = ScentReport.from_prediction(
            predicted_family=predicted_family,
            probabilities=probabilities.items(),
            intensity_index=intensity_from_reading(total_voc),
            environment=environment,
        )
        self.report = report

        self._render_chart(reading)
        self._update_result_panel(report)
        self.status_label.config(
            text=(
                f"Captured sample from '{profile_name}' → predicted as "
                f"{predicted_family}."
            )
        )

    # ------------------------------------------------------------------
    # UI update helpers
    # ------------------------------------------------------------------
    def _render_chart(self, reading: Dict[str, float]) -> None:
        self.chart_canvas.delete("all")

        values = [float(reading[feature]) for feature in VOC_FEATURES]
        max_value = max(itertools.chain(values, [1.0]))
        chart_height = self.CHART_HEIGHT - self.CHART_MARGIN * 2
        chart_width = self.CHART_WIDTH - self.CHART_MARGIN * 2
        bar_width = chart_width / max(len(values), 1)

        colors = ["#5b8ff9", "#61d9a1", "#65789b", "#f6bd16", "#7262fd", "#78d3f8"]

        for idx, feature in enumerate(VOC_FEATURES):
            value = float(reading[feature])
            height_ratio = value / max_value if max_value else 0
            bar_height = height_ratio * chart_height
            x0 = self.CHART_MARGIN + idx * bar_width + 10
            x1 = x0 + bar_width - 20
            y1 = self.CHART_HEIGHT - self.CHART_MARGIN
            y0 = y1 - bar_height
            color = colors[idx % len(colors)]
            self.chart_canvas.create_rectangle(x0, y0, x1, y1, fill=color, width=0)
            self.chart_canvas.create_text(
                (x0 + x1) / 2,
                y0 - 12,
                text=f"{value:.0f}",
                font=("Helvetica", 9),
                fill="#333333",
            )
            self.chart_canvas.create_text(
                (x0 + x1) / 2,
                y1 + 14,
                text=feature.replace("_ppb", "").replace("_", "\n"),
                font=("Helvetica", 9),
            )

        # Axes
        self.chart_canvas.create_line(
            self.CHART_MARGIN,
            self.CHART_HEIGHT - self.CHART_MARGIN,
            self.CHART_WIDTH - self.CHART_MARGIN,
            self.CHART_HEIGHT - self.CHART_MARGIN,
            width=2,
        )
        self.chart_canvas.create_line(
            self.CHART_MARGIN,
            self.CHART_HEIGHT - self.CHART_MARGIN,
            self.CHART_MARGIN,
            self.CHART_MARGIN,
            width=2,
        )

    def _update_metrics_panel(self) -> None:
        for child in self.metrics_panel.winfo_children():
            child.destroy()

        accuracy = self.metrics.get("overall_accuracy", 0.0)
        ttk.Label(
            self.metrics_panel,
            text=f"Overall accuracy: {accuracy:.2%}",
            style="Metrics.TLabel",
        ).pack(anchor=tk.W, padx=12, pady=(12, 4))

        per_class = self.metrics.get("per_class_accuracy", {})
        if per_class:
            table = ttk.Treeview(
                self.metrics_panel,
                columns=("label", "score"),
                show="headings",
                height=4,
            )
            table.heading("label", text="Scent family")
            table.heading("score", text="Accuracy")
            table.column("label", width=140)
            table.column("score", width=100, anchor=tk.CENTER)
            for label, score in per_class.items():
                display = "n/a" if score is None else f"{score:.2%}"
                table.insert("", tk.END, values=(label, display))
            table.pack(fill=tk.X, padx=12, pady=(0, 12))
        else:
            ttk.Label(
                self.metrics_panel,
                text="No evaluation metrics available.",
            ).pack(anchor=tk.W, padx=12, pady=(0, 12))

    def _update_result_panel(self, report: ScentReport) -> None:
        self.result_label.config(
            text=(
                f"Predicted scent: {report.predicted_family}\n"
                f"Confidence: {report.confidence:.1%}"
            )
        )

        self.intensity_var.set(report.intensity_index)
        self.intensity_label.config(text=f"{report.intensity_index:.1f}")

        for feature, var in self.environment_vars.items():
            value = report.environment.get(feature, "–")
            if isinstance(value, (int, float)):
                var.set(f"{value:.1f}")
            else:
                var.set(str(value))

        for item in self.probability_tree.get_children():
            self.probability_tree.delete(item)
        for label in self.artifacts.classes_:
            score = report.raw_probabilities.get(label)
            display = "n/a" if score is None else f"{score:.1%}"
            self.probability_tree.insert("", tk.END, values=(label, display))

def main() -> None:
    app = DigitalNoseApp()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()

