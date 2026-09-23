"""Unit tests for single and batch Neural Network implementations."""

import pytest
import numpy as np
from core.neural_network import NeuralNetwork, BatchNeuralNetwork
from core.genome import Genome
from config.settings import NetworkSettings


def test_neural_network_initialization():
    # Optimized default: 12 hidden neurons -> 130 weights
    settings = NetworkSettings()
    net = NeuralNetwork(settings=settings)

    assert net.weights_hidden.shape == (13, 7)
    assert net.weights_output.shape == (3, 13)
    assert settings.dna_length == 130

    # Legacy configuration: 6 hidden neurons -> 70 weights
    legacy_settings = NetworkSettings(num_hidden_neurons=6)
    legacy_net = NeuralNetwork(settings=legacy_settings)
    assert legacy_net.weights_hidden.shape == (7, 7)
    assert legacy_net.weights_output.shape == (3, 7)
    assert legacy_settings.dna_length == 70


def test_neural_network_forward_activation():
    net = NeuralNetwork()
    inputs = np.array([100.0, 32.0, 15.0, 33.0, 3.0, 15.0])
    outputs = net.forward(inputs)

    assert outputs.shape == (3,)


def test_neural_network_input_normalization():
    raw_inputs = np.array([400.0, 50.0, 50.0, 50.0, 4.0, 50.0])
    norm = NeuralNetwork.normalize_inputs(raw_inputs)
    assert norm.shape == (6,)
    assert np.all(norm >= 0.0)
    assert np.all(norm <= 1.5)

    # Test batch normalization
    batch_raw = np.array([[800.0, 100.0, 100.0, 100.0, 8.0, 100.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    batch_norm = NeuralNetwork.normalize_inputs(batch_raw)
    assert batch_norm.shape == (2, 6)
    assert np.allclose(batch_norm[0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(batch_norm[1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_batch_and_single_equivalence():
    """Validates that BatchNeuralNetwork produces identical numerical results to individual NeuralNetworks."""
    P = 10
    settings = NetworkSettings()
    genomes = [Genome.random(settings.dna_length, -500, 500) for _ in range(P)]

    batch_net = BatchNeuralNetwork(population_size=P, settings=settings)
    batch_net.load_population_genomes(genomes)

    # Random batch inputs
    batch_inputs = np.random.uniform(0, 200, size=(P, 6))

    batch_outputs = batch_net.forward_batch(batch_inputs)

    for i in range(P):
        single_net = NeuralNetwork(settings=settings, genome=genomes[i])
        single_out = single_net.forward(batch_inputs[i])

        np.testing.assert_allclose(
            batch_outputs[i],
            single_out,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Discrepancy found at individual {i}"
        )
