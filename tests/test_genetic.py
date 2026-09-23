"""Unit tests for Genetic Algorithm operations, elitism, and decay."""

import pytest
import numpy as np
from core.genetic_algorithm import GeneticAlgorithm
from core.genome import Genome
from config.settings import GeneticSettings, NetworkSettings


def test_genetic_algorithm_initialization():
    ga = GeneticAlgorithm(population_size=100)
    assert len(ga.population) == 100
    assert ga.population[0].length == ga.dna_length
    assert ga.generation == 0
    assert ga.range_random == 70.0


def test_genetic_elitism_preservation():
    ga = GeneticAlgorithm(population_size=50)

    # Assign distinct fitnesses
    fitnesses = np.arange(50, dtype=np.float64)  # Individual 49 has highest fitness 49.0
    champion_before = ga.population[49].copy()

    best_fit, avg_fit, champion = ga.evolve(fitnesses)

    assert best_fit == 49.0
    # Champion at index 0 must strictly match champion_before
    np.testing.assert_array_equal(ga.population[0].weights, champion_before.weights)


def test_range_random_decay():
    settings = GeneticSettings(
        population_size=20,
        initial_range_random=50.0,
        decay_rate=0.9,
        min_range_random=20.0
    )
    ga = GeneticAlgorithm(population_size=20, settings=settings)

    fitnesses = np.ones(20, dtype=np.float64)

    # Step 1: 50 * 0.9 = 45.0
    ga.evolve(fitnesses)
    assert abs(ga.range_random - 45.0) < 1e-4

    # Many steps with increasing fitness: should clamp at min_range_random (20.0)
    for i in range(1, 31):
        ga.evolve(np.ones(20, dtype=np.float64) * (10.0 + i))

    assert ga.range_random == 20.0
