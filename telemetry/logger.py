"""Telemetry Logger recording per-generation statistics into CSV format."""

import csv
import time
from pathlib import Path
from typing import Optional


class TelemetryLogger:
    """Logs evolutionary training telemetry to a CSV file."""

    def __init__(self, log_dir: Optional[Path] = None, filename: str = "training_stats.csv"):
        self.log_dir = log_dir or Path("telemetry")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.start_time = time.perf_counter()

        self._init_file()

    def _init_file(self) -> None:
        """Initializes CSV header if file does not exist."""
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "generation",
                    "best_fitness",
                    "avg_fitness",
                    "std_fitness",
                    "record_distance",
                    "dead_count",
                    "population_size",
                    "elapsed_seconds",
                ])

    def log_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        std_fitness: float,
        record_distance: float,
        dead_count: int,
        population_size: int,
    ) -> None:
        """Appends one generation entry to the CSV log."""
        elapsed = time.perf_counter() - self.start_time
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                f"{best_fitness:.2f}",
                f"{avg_fitness:.2f}",
                f"{std_fitness:.2f}",
                f"{record_distance:.2f}",
                dead_count,
                population_size,
                f"{elapsed:.2f}",
            ])
