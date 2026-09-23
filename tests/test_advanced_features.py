"""Automated unit tests for advanced features: Crossover, Sound, Telemetry, and Versus mode."""

import pytest
import numpy as np
from pathlib import Path

from config.settings import GameConfig, DinoState
from core.genetic_algorithm import GeneticAlgorithm
from core.genome import Genome
from game.engine import GameEngine
from rendering.audio import SoundSynthesizer
from telemetry.logger import TelemetryLogger
from telemetry.plotter import TelemetryPlotter


def test_genetic_crossover():
    ga = GeneticAlgorithm(population_size=10, crossover_rate=1.0)
    p1 = Genome(np.zeros(70))
    p2 = Genome(np.ones(70))

    # Crossover must produce blended / mixed values
    child = ga._crossover(p1, p2)
    assert child.shape == (70,)
    # Child elements must be in [0, 1]
    assert np.all(child >= 0.0)
    assert np.all(child <= 1.0)
    # Child must not be completely empty
    assert np.any(child > 0.0)


def test_adaptive_hypermutation():
    ga = GeneticAlgorithm(population_size=10)
    initial_range = ga.range_random

    # Simulate 6 generations of stagnation (constant fitness)
    stagnant_fitness = np.ones(10, dtype=np.float64) * 100.0

    # First gen sets record
    ga.evolve(stagnant_fitness)
    assert ga.stagnation_counter == 0
    assert ga.hypermutation_active is False

    # Next 5 gens without record increase
    for _ in range(5):
        ga.evolve(stagnant_fitness)

    assert ga.stagnation_counter >= 5
    assert ga.hypermutation_active is True


def test_sound_synthesizer():
    synth = SoundSynthesizer()
    synth.initialize()

    assert synth.initialized is True
    assert synth.muted is False

    # Test mute toggle
    muted = synth.toggle_mute()
    assert muted is True
    assert synth.muted is True

    synth.toggle_mute()
    assert synth.muted is False

    # Test playback triggers without crashing
    synth.play_jump()
    synth.play_airplane()
    synth.play_death()
    synth.play_score()


def test_telemetry_logger_and_plotter(tmp_path: Path):
    log_file = tmp_path / "test_stats.csv"
    logger = TelemetryLogger(log_dir=tmp_path, filename="test_stats.csv")

    for gen in range(1, 4):
        logger.log_generation(
            generation=gen,
            best_fitness=100.0 * gen,
            avg_fitness=50.0 * gen,
            std_fitness=10.0,
            record_distance=200.0 * gen,
            dead_count=100,
            population_size=100,
        )

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "generation,best_fitness" in content
    assert "3,300.00" in content

    # Test plot generation
    plotter = TelemetryPlotter(csv_path=log_file)
    plot_file = tmp_path / "test_report.png"
    out_img = plotter.generate_report(output_image=plot_file)
    assert out_img is not None
    assert plot_file.exists()
    assert plot_file.stat().st_size > 1000


def test_versus_mode_engine():
    config = GameConfig()
    engine = GameEngine(config=config, mode="versus")

    assert engine.population_size == 2
    assert len(engine.dinosaurs) == 2

    dino_human = engine.dinosaurs[0]
    dino_ai = engine.dinosaurs[1]

    assert dino_human.color_idx == 2  # Green
    assert dino_ai.color_idx == 4     # Blue

    # Step simulation with human jump input
    finished = engine.step(dt=0.005, player_inputs=(True, False, False))
    assert finished is False
    assert dino_human.state == DinoState.JUMPING
