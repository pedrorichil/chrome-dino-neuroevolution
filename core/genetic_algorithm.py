"""Genetic Algorithm engine with elitism, crossover, adaptive hypermutation, and statistics."""

import random
from typing import List, Tuple, Optional
import numpy as np
from config.settings import GeneticSettings, NetworkSettings
from core.genome import Genome


class GeneticAlgorithm:
    """
    Advanced Genetic Algorithm:
    - O(N log N) vectorized sorting
    - Strict Elitism (champion preserved unchanged)
    - Sexual reproduction / Crossover (Uniform & Arithmetic) between top performers
    - Adaptive Hypermutation: detects stagnation (5 gens without improvement) and bursts mutations
    - Historical metrics tracking
    """

    def __init__(
        self,
        population_size: int = 2000,
        settings: Optional[GeneticSettings] = None,
        net_settings: Optional[NetworkSettings] = None,
        crossover_rate: float = 0.5,
    ):
        self.population_size = population_size
        self.settings = settings or GeneticSettings(population_size=population_size)
        self.net_settings = net_settings or NetworkSettings()
        self.dna_length = self.net_settings.dna_length
        self.crossover_rate = crossover_rate

        self.range_random: float = self.settings.initial_range_random
        self.generation: int = 0

        # Stagnation tracking for adaptive hypermutation
        self.highest_record: float = 0.0
        self.stagnation_counter: int = 0
        self.hypermutation_active: bool = False

        # Population genomes: List of Genome instances
        self.population: List[Genome] = [
            Genome.random(self.dna_length, self.settings.random_weight_min, self.settings.random_weight_max)
            for _ in range(self.population_size)
        ]

        # Historical metrics
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.std_fitness_history: List[float] = []

    def seed_individual(self, index: int, genome: Genome) -> None:
        """Seeds a specific individual with a given genome."""
        if index < self.population_size:
            self.population[index] = genome.copy()

    def seed_all_from_best(self, best_genome: Genome) -> None:
        """Seeds the entire population as clones/mutations of a single champion genome."""
        if best_genome.length != self.dna_length:
            self.dna_length = best_genome.length
        self.population[0] = best_genome.copy()
        for i in range(1, self.population_size):
            self.population[i] = best_genome.copy()

    def _crossover(self, parent1: Genome, parent2: Genome) -> np.ndarray:
        """Combines two parent genomes using uniform and arithmetic crossover."""
        method = random.choice(["uniform", "arithmetic"])
        length = len(parent1.weights)
        if method == "uniform":
            # Mask selects genes randomly from parent 1 or parent 2
            mask = np.random.rand(length) < 0.5
            return np.where(mask, parent1.weights, parent2.weights)
        else:
            # Arithmetic blend crossover
            alpha = random.uniform(0.2, 0.8)
            return alpha * parent1.weights + (1.0 - alpha) * parent2.weights

    def evolve(self, fitness_scores: np.ndarray) -> Tuple[float, float, Genome]:
        """
        Executes one evolutionary step:
        1. Records generation metrics (Best, Average, Std Dev).
        2. Detects stagnation and activates Adaptive Hypermutation if stalled.
        3. Sorts population by fitness descending.
        4. Preserves champion (Elitism).
        5. Generates children via Crossover (top 10% parents) and Champion Cloning.
        6. Applies stochastic mutations.
        Returns: (best_fitness, avg_fitness, champion_genome)
        """
        fitness_arr = np.asarray(fitness_scores, dtype=np.float64)
        best_fitness = float(np.max(fitness_arr))
        avg_fitness = float(np.mean(fitness_arr))
        std_fitness = float(np.std(fitness_arr))

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.std_fitness_history.append(std_fitness)

        # Check for stagnation
        if best_fitness > self.highest_record:
            self.highest_record = best_fitness
            self.stagnation_counter = 0
            self.hypermutation_active = False
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= 5:
                # Trigger adaptive hypermutation burst
                self.hypermutation_active = True
                self.range_random = min(self.dna_length, self.range_random * 1.5)

        # Sort indices descending by fitness
        sorted_indices = np.argsort(-fitness_arr)
        champion_idx = sorted_indices[0]
        champion_genome = self.population[champion_idx].copy()

        # Top 10% individuals eligible for breeding
        top_k = max(2, int(self.population_size * 0.10))
        top_parents = [self.population[idx] for idx in sorted_indices[:top_k]]

        # Elitism: champion always preserved at index 0
        new_population: List[Genome] = [champion_genome.copy()]

        # Mutation bounds
        max_mutations = max(1, int(self.range_random))
        if self.hypermutation_active:
            max_mutations = min(self.dna_length, int(max_mutations * 1.5))

        # Build offspring for indices 1 to P-1
        for _ in range(1, self.population_size):
            if random.random() < self.crossover_rate and len(top_parents) >= 2:
                # Crossover between two randomly selected top performers
                p1, p2 = random.sample(top_parents, 2)
                child_weights = self._crossover(p1, p2)
            else:
                # Clone champion
                child_weights = champion_genome.weights.copy()

            # Apply mutations (exploration + fine Gaussian exploitation)
            num_mutations = random.randint(1, max_mutations)
            mut_indices = np.random.randint(0, self.dna_length, size=num_mutations)
            mut_types = np.random.randint(0, 4, size=num_mutations)
            sigma = getattr(self.settings, "gaussian_mutation_sigma", 3.0)

            for idx, m_type in zip(mut_indices, mut_types):
                if m_type == 0:
                    child_weights[idx] = random.uniform(
                        self.settings.random_weight_min, self.settings.random_weight_max
                    )
                elif m_type == 1:
                    child_weights[idx] *= random.uniform(0.7, 1.3)
                elif m_type == 2:
                    child_weights[idx] += random.uniform(-10.0, 10.0)
                else:
                    child_weights[idx] += np.random.normal(0.0, sigma)

            new_population.append(Genome(child_weights))

        self.population = new_population

        # Decay range_random if not hypermutating
        if not self.hypermutation_active:
            self.range_random = max(self.settings.min_range_random, self.range_random * self.settings.decay_rate)

        self.generation += 1
        return best_fitness, avg_fitness, champion_genome
