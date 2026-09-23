"""Telemetry Plotter generating publication-quality analytical graphs with Matplotlib."""

from pathlib import Path
from typing import Optional
import csv
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class TelemetryPlotter:
    """Generates charts from training telemetry data."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or Path("telemetry/training_stats.csv")

    def generate_report(self, output_image: Optional[Path] = None) -> Optional[Path]:
        """Reads CSV and outputs a multi-panel analysis PNG image."""
        if not self.csv_path.exists():
            print(f"[Aviso] Arquivo de telemetria não encontrado: {self.csv_path}")
            return None

        generations = []
        bests = []
        avgs = []
        stds = []
        records = []
        times = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                generations.append(int(row["generation"]))
                bests.append(float(row["best_fitness"]))
                avgs.append(float(row["avg_fitness"]))
                stds.append(float(row["std_fitness"]))
                records.append(float(row["record_distance"]))
                times.append(float(row["elapsed_seconds"]))

        if len(generations) == 0:
            return None

        out_path = output_image or Path("telemetry/evolution_report.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Google Dino AI - Relatório Analítico de Evolução", fontsize=14, fontweight="bold")

        # Top Plot: Best, Mean & Std Dev
        gens = np.array(generations)
        b_arr = np.array(bests)
        a_arr = np.array(avgs)
        s_arr = np.array(stds)

        ax1.plot(gens, b_arr, label="Melhor Fitness", color="#1e90ff", linewidth=2.5)
        ax1.plot(gens, a_arr, label="Média Populacional", color="#dc143c", linewidth=2.0)
        ax1.fill_between(gens, np.maximum(0, a_arr - s_arr), a_arr + s_arr, color="#dc143c", alpha=0.15, label="±1 Desvio Padrão")
        ax1.set_ylabel("Fitness (Pontuação)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left")

        # Bottom Plot: Record Distance
        r_arr = np.array(records)
        ax2.plot(gens, r_arr, label="Distância Recorde (pixels)", color="#2e8b57", linewidth=2.0)
        ax2.set_xlabel("Geração", fontsize=11)
        ax2.set_ylabel("Distância Recorde (px)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[Sucesso] Relatório analítico gerado em: {out_path}")
        return out_path
