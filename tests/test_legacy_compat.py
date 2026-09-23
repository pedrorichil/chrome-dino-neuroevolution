"""Unit tests for Genome serialization and legacy model compatibility."""

import os
from pathlib import Path
import pytest
import numpy as np
from core.genome import Genome
from core.neural_network import NeuralNetwork


def test_genome_json_roundtrip(tmp_path: Path):
    weights = np.random.uniform(-100, 100, size=70)
    g = Genome(weights)

    out_file = tmp_path / "test_model.json"
    g.to_json(out_file, metadata={"creator": "test"})

    loaded = Genome.from_json(out_file)
    np.testing.assert_allclose(loaded.weights, g.weights, rtol=1e-6)


def test_genome_legacy_binary_roundtrip(tmp_path: Path):
    weights = np.random.uniform(-1000, 1000, size=70)
    g = Genome(weights)

    out_file = tmp_path / "test_model.bin"
    g.to_legacy_binary(out_file)

    loaded = Genome.from_legacy_binary(out_file)
    assert loaded.length == 70
    np.testing.assert_allclose(loaded.weights, g.weights, rtol=1e-9)


def test_load_existing_legacy_x1_model():
    legacy_file = Path("models/rede_legacy_x1")
    if not legacy_file.exists():
        pytest.skip("Legacy model file not present in models/")

    genome = Genome.from_legacy_binary(legacy_file)
    assert genome.length == 70

    net = NeuralNetwork(genome=genome)
    # Give input where obstacle is close
    inputs = np.array([50.0, 32.0, 15.0, 33.0, 3.0, 15.0])
    outputs = net.forward(inputs)

    assert len(outputs) == 3
    # Check that network produces numeric action
    assert np.any(outputs > 0)
