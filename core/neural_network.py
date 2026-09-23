"""Neural Network implementation with single and batch inference using NumPy."""

from typing import Tuple, Optional, Union
import numpy as np
from config.settings import NetworkSettings
from core.genome import Genome


class NeuralNetwork:
    """
    Multilayer Perceptron (MLP) with configurable topology, LeakyReLU activations,
    and input normalization.
    - Default: 6 inputs (+1 bias) -> 12 hidden neurons (+1 bias) -> 3 outputs.
    - Supports LeakyReLU to prevent dying neurons and retain gradient/expressiveness.
    """

    def __init__(self, settings: Optional[NetworkSettings] = None, genome: Optional[Genome] = None):
        if settings is None and genome is not None and genome.length == 70:
            settings = NetworkSettings(num_hidden_neurons=6)
        self.settings = settings or NetworkSettings()

        self.num_active_hidden = self.settings.num_hidden_neurons
        self.hidden_dim = self.settings.total_hidden
        self.input_dim = self.settings.total_inputs
        self.output_dim = self.settings.num_outputs
        self.alpha = getattr(self.settings, "leaky_alpha", 0.05)

        # Weights storage
        # Hidden layer weights: shape (hidden_dim, input_dim)
        # Output layer weights: shape (output_dim, hidden_dim)
        self.weights_hidden = np.zeros((self.hidden_dim, self.input_dim), dtype=np.float64)
        self.weights_output = np.zeros((self.output_dim, self.hidden_dim), dtype=np.float64)

        # Activations cached for visualizer
        self.last_inputs = np.zeros(self.input_dim, dtype=np.float64)
        self.last_hidden = np.zeros(self.hidden_dim, dtype=np.float64)
        self.last_outputs = np.zeros(self.output_dim, dtype=np.float64)

        if genome is not None:
            self.load_genome(genome)

    @staticmethod
    def normalize_inputs(inputs: np.ndarray) -> np.ndarray:
        """
        Normalizes raw physical sensor inputs into well-conditioned [0.0, 1.0] ranges.
        Supports single vector shape (6,) or batch shape (P, 6).
        Features:
        0: Distance (0 to 800) -> [0.0, 1.0]
        1: Obstacle Width (0 to 100) -> [0.0, 1.0]
        2: Obstacle Y (0 to 100) -> [0.0, 1.0]
        3: Obstacle Height (0 to 100) -> [0.0, 1.0]
        4: Speed (3.0 to 8.0) -> [0.0, 1.0]
        5: Dino Y (15.0 to 100.0) -> [0.0, 1.0]
        """
        scale = np.array([800.0, 100.0, 100.0, 100.0, 8.0, 100.0], dtype=np.float64)
        norm = inputs / scale
        return np.clip(norm, 0.0, 1.5)

    def load_genome(self, genome: Genome) -> None:
        """Unpacks 1D DNA into weight matrices matching memory layout."""
        if genome.length != self.settings.dna_length:
            if genome.length == 70:
                self.settings = NetworkSettings(num_hidden_neurons=6)
                self.num_active_hidden = 6
                self.hidden_dim = 7
                self.weights_hidden = np.zeros((7, 7), dtype=np.float64)
                self.weights_output = np.zeros((self.output_dim, 7), dtype=np.float64)
            else:
                raise ValueError(
                    f"Genome length {genome.length} does not match expected {self.settings.dna_length}"
                )
        hidden_count = self.hidden_dim * self.input_dim
        self.weights_hidden = genome.weights[:hidden_count].reshape((self.hidden_dim, self.input_dim))
        self.weights_output = genome.weights[hidden_count:].reshape((self.output_dim, self.hidden_dim))

    def extract_genome(self) -> Genome:
        """Flattens weight matrices back into a 1D Genome."""
        flat = np.concatenate([self.weights_hidden.flatten(), self.weights_output.flatten()])
        return Genome(flat)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Applies LeakyReLU activation."""
        if getattr(self.settings, "activation", "leaky_relu") == "leaky_relu":
            return np.where(x > 0.0, x, self.alpha * x)
        return np.maximum(0.0, x)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculates forward pass for a single entity:
        inputs: array-like of length 6 (distance, width, height, length, speed, dino_y).
        Returns: array of shape (3,) [Jump, Duck, Airplane].
        """
        # Ensure input array with bias = 1.0
        if len(inputs) == self.settings.num_inputs:
            full_inputs = np.empty(self.input_dim, dtype=np.float64)
            full_inputs[: self.settings.num_inputs] = inputs
            full_inputs[self.settings.num_inputs] = 1.0
        else:
            full_inputs = np.asarray(inputs, dtype=np.float64)

        self.last_inputs = full_inputs

        # Compute active hidden neurons
        raw_hidden = self.weights_hidden[: self.num_active_hidden, :] @ full_inputs
        activated_hidden = self._activate(raw_hidden)

        # Full hidden layer has active neurons + bias neuron (value 1.0)
        full_hidden = np.empty(self.hidden_dim, dtype=np.float64)
        full_hidden[: self.num_active_hidden] = activated_hidden
        full_hidden[self.num_active_hidden] = 1.0

        self.last_hidden = full_hidden

        # Compute outputs
        raw_output = self.weights_output @ full_hidden
        self.last_outputs = self._activate(raw_output)

        return self.last_outputs


class BatchNeuralNetwork:
    """
    Vectorized high-performance batch inference for an entire population.
    Handles P dinosaurs simultaneously via NumPy tensor operations.
    """

    def __init__(self, population_size: int, settings: Optional[NetworkSettings] = None):
        self.population_size = population_size
        self.settings = settings or NetworkSettings()

        self.num_active_hidden = self.settings.num_hidden_neurons
        self.hidden_dim = self.settings.total_hidden
        self.input_dim = self.settings.total_inputs
        self.output_dim = self.settings.num_outputs
        self.alpha = getattr(self.settings, "leaky_alpha", 0.05)

        # Tensor shapes: (P, num_active_hidden, input_dim) and (P, output_dim, hidden_dim)
        self.weights_hidden = np.zeros((population_size, self.num_active_hidden, self.input_dim), dtype=np.float64)
        self.weights_output = np.zeros((population_size, self.output_dim, self.hidden_dim), dtype=np.float64)

        # Pre-allocated arrays for memory efficiency
        self._hidden_buffer = np.ones((population_size, self.hidden_dim), dtype=np.float64)
        self._inputs_buffer = np.ones((population_size, self.input_dim), dtype=np.float64)

    def load_population_genomes(self, genomes: list[Genome]) -> None:
        """Loads weights for all individuals from a list of Genomes."""
        hidden_active_len = self.num_active_hidden * self.input_dim
        hidden_total_len = self.hidden_dim * self.input_dim

        for i, g in enumerate(genomes):
            self.weights_hidden[i] = g.weights[:hidden_active_len].reshape((self.num_active_hidden, self.input_dim))
            self.weights_output[i] = g.weights[hidden_total_len:].reshape((self.output_dim, self.hidden_dim))

    def update_single_genome(self, index: int, genome: Genome) -> None:
        """Updates weights for a specific individual in the batch."""
        hidden_active_len = self.num_active_hidden * self.input_dim
        hidden_total_len = self.hidden_dim * self.input_dim

        self.weights_hidden[index] = genome.weights[:hidden_active_len].reshape((self.num_active_hidden, self.input_dim))
        self.weights_output[index] = genome.weights[hidden_total_len:].reshape((self.output_dim, self.hidden_dim))

    def forward_batch(self, batch_inputs: np.ndarray, alive_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes outputs for all alive individuals in batch.
        batch_inputs: shape (P, 6)
        alive_mask: boolean array of shape (P,)
        Returns: shape (P, 3) outputs
        """
        self._inputs_buffer[:, :6] = batch_inputs
        self._inputs_buffer[:, 6] = 1.0  # Bias

        alpha = self.alpha
        is_leaky = getattr(self.settings, "activation", "leaky_relu") == "leaky_relu"

        if alive_mask is not None and not np.all(alive_mask):
            indices = np.where(alive_mask)[0]
            if len(indices) == 0:
                return np.zeros((self.population_size, self.output_dim), dtype=np.float64)

            inp = self._inputs_buffer[indices]
            w_h = self.weights_hidden[indices]
            w_o = self.weights_output[indices]

            # Einsum: pk = sum_i(inp_pi * w_h_pki)
            raw_h = np.einsum('pi,pki->pk', inp, w_h)
            h_active = np.where(raw_h > 0.0, raw_h, alpha * raw_h) if is_leaky else np.maximum(0.0, raw_h)
            h_full = np.pad(h_active, ((0, 0), (0, 1)), constant_values=1.0)

            raw_o = np.einsum('pi,poi->po', h_full, w_o)
            out_subset = np.where(raw_o > 0.0, raw_o, alpha * raw_o) if is_leaky else np.maximum(0.0, raw_o)

            full_out = np.zeros((self.population_size, self.output_dim), dtype=np.float64)
            full_out[indices] = out_subset
            return full_out
        else:
            raw_h = np.einsum('pi,pki->pk', self._inputs_buffer, self.weights_hidden)
            h_active = np.where(raw_h > 0.0, raw_h, alpha * raw_h) if is_leaky else np.maximum(0.0, raw_h)
            self._hidden_buffer[:, :self.num_active_hidden] = h_active
            self._hidden_buffer[:, self.num_active_hidden] = 1.0

            raw_o = np.einsum('pi,poi->po', self._hidden_buffer, self.weights_output)
            return np.where(raw_o > 0.0, raw_o, alpha * raw_o) if is_leaky else np.maximum(0.0, raw_o)
