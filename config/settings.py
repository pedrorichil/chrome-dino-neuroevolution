"""Configuration module for Google Dino AI."""

from dataclasses import dataclass, field
from typing import Tuple, List
import enum


class DinoState(enum.IntEnum):
    """Enumeration of possible dinosaur states."""
    RUNNING = 0
    DUCKING = 1
    JUMPING = 2
    DEAD = 3
    FLYING = 4


class ObstacleType(enum.IntEnum):
    """Enumeration of obstacle types."""
    CACTUS_SMALL_1 = 0
    CACTUS_SMALL_2 = 1
    CACTUS_SMALL_3 = 2
    CACTUS_LARGE_1 = 3
    CACTUS_LARGE_2 = 4
    BIRD = 5
    SPIKES = 6


@dataclass(frozen=True)
class DisplaySettings:
    """Window and rendering parameters."""
    screen_width: int = 1366
    screen_height: int = 768
    title: str = "Google Dinosaur AI (Python Edition)"
    fps: int = 60
    graph_width: int = 600
    graph_height: int = 350
    colors: Tuple[Tuple[int, int, int], ...] = (
        (128, 128, 128),  # Cinza
        (255, 215, 0),    # Amarelo
        (34, 139, 34),    # Verde
        (220, 20, 60),    # Vermelho
        (30, 144, 255),   # Azul
        (0, 206, 209),    # Ciano
        (255, 140, 0),    # Laranja
        (138, 43, 226),   # Roxo
    )


@dataclass(frozen=True)
class PhysicsSettings:
    """Physics simulation constants mirroring original C dynamics."""
    gravity: float = 0.08
    jump_impulse: float = 4.0
    duck_fall_speed: float = 2.0
    ground_y: float = 15.0
    initial_speed: float = -3.0
    max_speed_magnitude: float = 8.0
    speed_acceleration: float = 0.0005
    airplane_distance_max: float = 820.0
    airplane_cooldown_initial: float = 4000.0
    hitbox_horizontal_correction: int = 7
    hitbox_vertical_correction: int = 5
    dino_initial_x_range: Tuple[int, int] = (200, 400)


@dataclass(frozen=True)
class NetworkSettings:
    """Neural network topology configuration."""
    num_inputs: int = 6
    num_hidden_layers: int = 1
    num_hidden_neurons: int = 12
    num_outputs: int = 3
    bias: int = 1
    activation: str = "leaky_relu"
    leaky_alpha: float = 0.05

    @property
    def total_inputs(self) -> int:
        return self.num_inputs + self.bias

    @property
    def total_hidden(self) -> int:
        return self.num_hidden_neurons + self.bias

    @property
    def dna_length(self) -> int:
        """
        Architecture allocation:
        Hidden layer 0: total_hidden * total_inputs
        Output layer: num_outputs * total_hidden
        For 12 hidden: (13 * 7) + (3 * 13) = 91 + 39 = 130 weights.
        """
        return (self.total_hidden * self.total_inputs) + (self.num_outputs * self.total_hidden)


@dataclass
class GeneticSettings:
    """Genetic algorithm hyperparameters."""
    population_size: int = 2000
    initial_range_random: float = 70.0
    decay_rate: float = 0.99
    min_range_random: float = 20.0
    random_weight_min: float = -1000.0
    random_weight_max: float = 1000.0
    fitness_per_running_tick: float = 2.0
    fitness_per_air_tick: float = 1.0
    bonus_obstacle_cleared: float = 50.0
    penalty_idle_jump: float = 0.5
    gaussian_mutation_sigma: float = 3.0


@dataclass
class GameConfig:
    """Master game configuration container."""
    display: DisplaySettings = field(default_factory=DisplaySettings)
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    genetic: GeneticSettings = field(default_factory=GeneticSettings)
