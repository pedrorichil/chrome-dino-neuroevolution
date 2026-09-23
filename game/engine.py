"""Headless Game Engine orchestrating physics, entities, sensors, and simulation loop."""

import random
from typing import List, Optional, Tuple, Callable
import numpy as np

from config.settings import GameConfig, DinoState, ObstacleType
from game.entities.dinosaur import Dinosaur
from game.entities.obstacle import Obstacle
from game.entities.environment import Environment
from game.physics import check_dino_obstacle_collision
from game.spawner import ObstacleSpawner
from core.genetic_algorithm import GeneticAlgorithm
from core.neural_network import BatchNeuralNetwork, NeuralNetwork
from core.genome import Genome
from telemetry.logger import TelemetryLogger


class GameEngine:
    """
    Decoupled game engine capable of running in Headless mode (no graphics)
    at tens of thousands of FPS, or synchronized with a graphical renderer.
    Supports 'training', 'playable', 'evaluation', and 'versus' modes.
    """

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        mode: str = "training",
        enable_telemetry: bool = False,
    ):
        self.config = config or GameConfig()
        self.mode = mode  # "training", "playable", "evaluation", "versus"
        self.enable_telemetry = enable_telemetry

        if self.mode == "training":
            self.population_size = self.config.genetic.population_size
        elif self.mode == "versus":
            self.population_size = 2
        else:
            self.population_size = 1

        # Spawner and environment
        self.spawner = ObstacleSpawner()
        self.environment = Environment(self.config.display)

        # Dinosaurs
        self.dinosaurs: List[Dinosaur] = []
        for i in range(self.population_size):
            if self.mode == "versus":
                # Dino 0: Human (Green, index 2), Dino 1: AI (Blue, index 4)
                color = 2 if i == 0 else 4
                init_x = 260.0 if i == 0 else 340.0
            else:
                color = i % 8
                init_x = 300.0 + (random.randint(-100, 100) if self.mode == "training" else 0.0)

            self.dinosaurs.append(
                Dinosaur(
                    index=i,
                    x=init_x,
                    y=self.config.physics.ground_y,
                    color_idx=color,
                )
            )

        # Active obstacles pool (max 7 active on screen)
        self.obstacles: List[Obstacle] = []

        # Simulation metrics
        self.speed = self.config.physics.initial_speed
        self.distance = 0.0
        self.record_distance = 0.0
        self.dead_count = 0
        self.best_dinosaur: Dinosaur = self.dinosaurs[0]

        # Sound events triggered in current step ('jump', 'airplane', 'death', 'score')
        self.sound_events: List[str] = []

        # AI systems
        self.ga: Optional[GeneticAlgorithm] = None
        self.batch_brain: Optional[BatchNeuralNetwork] = None
        self.single_brain: Optional[NeuralNetwork] = None

        if self.mode == "training":
            self.ga = GeneticAlgorithm(
                population_size=self.population_size,
                settings=self.config.genetic,
                net_settings=self.config.network,
            )
            self.batch_brain = BatchNeuralNetwork(
                population_size=self.population_size,
                settings=self.config.network,
            )
            self.batch_brain.load_population_genomes(self.ga.population)
        else:
            self.single_brain = NeuralNetwork(self.config.network)

        # Telemetry
        self.telemetry_logger: Optional[TelemetryLogger] = (
            TelemetryLogger() if (self.enable_telemetry and self.mode == "training") else None
        )

        self.start_new_match()

    def start_new_match(self) -> None:
        """Initializes a new match / generation."""
        self.spawner.generate_track()
        self.obstacles = self.spawner.spawn_initial_pool(max_active=7)

        self.speed = self.config.physics.initial_speed
        self.distance = 0.0
        self.dead_count = 0

        for i, dino in enumerate(self.dinosaurs):
            if self.mode == "versus":
                init_x = 260.0 if i == 0 else 340.0
            else:
                init_x = 300.0 + (random.randint(-100, 100) if self.mode == "training" else 0.0)
            dino.reset(x=init_x, y=self.config.physics.ground_y)

        self.best_dinosaur = self.dinosaurs[0]

    def find_next_obstacle(self, dino_x: float) -> Optional[Obstacle]:
        """Finds the nearest obstacle ahead of dino_x where (obs.x + obs.width > dino_x)."""
        closest_obs = None
        min_dist = float("inf")

        for obs in self.obstacles:
            if obs.x + obs.width > dino_x:
                if obs.x < min_dist:
                    min_dist = obs.x
                    closest_obs = obs

        return closest_obs or (self.obstacles[0] if self.obstacles else None)

    def get_sensors_for_dino(self, dino: Dinosaur) -> np.ndarray:
        """Extracts sensory inputs for a dinosaur."""
        obs = self.find_next_obstacle(dino.x)
        if obs is None:
            return np.array([9999.0, 0.0, 0.0, 0.0, abs(self.speed), dino.y], dtype=np.float64)

        return np.array([
            obs.x - dino.x,
            float(obs.width),
            float(obs.y),
            float(obs.height),
            abs(self.speed),
            float(dino.y),
        ], dtype=np.float64)

    @staticmethod
    def _arbitrate_actions(
        jump_val: float,
        duck_val: float,
        plane_val: float,
        cooldown: float,
        threshold: float = 0.15,
        speed_mag: float = 3.0,
    ) -> Tuple[bool, bool, bool]:
        """
        High-precision motor arbitration engine:
        - Eliminates jump/duck conflicts.
        - Dynamic hysteresis: adjusts firing threshold slightly based on speed to compensate for reaction frames.
        - Soft-deadzone filtering: prevents erratic flapping/micro-decision spasms.
        """
        jump = False
        duck = False
        airplane = False

        # Dynamic threshold tuning: as speed grows, lower threshold by up to 0.05 for sharper reaction time
        speed_ratio = min(1.0, max(0.0, (speed_mag - 3.0) / 5.0))
        effective_threshold = threshold - (0.05 * speed_ratio)

        # Confidence margin (hysteresis) to avoid jitter when outputs are nearly tied
        hysteresis_margin = 0.03

        if plane_val > effective_threshold and plane_val > (jump_val + hysteresis_margin) and cooldown <= 0:
            airplane = True
        elif jump_val > effective_threshold and jump_val >= (duck_val - hysteresis_margin):
            jump = True
        elif duck_val > effective_threshold:
            duck = True

        return jump, duck, airplane

    def step(self, dt: float = 0.005, player_inputs: Optional[Tuple[bool, bool, bool]] = None) -> bool:
        """
        Executes one physics tick of the simulation.
        Returns True if the generation/match just ended and reset, False otherwise.
        """
        self.sound_events.clear()
        speed_mag = abs(self.speed)
        penalty_idle = getattr(self.config.genetic, "penalty_idle_jump", 0.5)
        bonus_cleared = getattr(self.config.genetic, "bonus_obstacle_cleared", 50.0)

        # 1. Update environment
        self.environment.update(self.speed)

        # 2. Update obstacles and recycle off-screen
        bonus_apex = getattr(self.config.genetic, "bonus_apex_precision", 30.0)
        for i, obs in enumerate(self.obstacles):
            old_right = obs.x + obs.width
            obs.update(self.speed, dt=dt)
            new_right = obs.x + obs.width

            # Reward dinosaurs that successfully cleared this obstacle
            for dino in self.dinosaurs:
                if dino.is_alive and old_right >= dino.x and new_right < dino.x:
                    dino.fitness += bonus_cleared
                    # Bônus de ápice: se o dinossauro passou voando por cima com margem perfeita (sem salto afobado)
                    if dino.state in (DinoState.JUMPING, DinoState.FLYING):
                        clearance = (dino.y) - (obs.y + obs.height)
                        if 0.0 <= clearance <= 35.0:
                            # Passou raspando cirurgicamente no topo
                            dino.fitness += bonus_apex

            if obs.x + obs.width < -20.0:
                self.obstacles[i] = self.spawner.get_next_obstacle(self.distance)

        # 3. Update AI / Player decisions
        if self.mode == "training" and self.batch_brain:
            alive_mask = np.array([d.is_alive for d in self.dinosaurs], dtype=bool)
            if np.any(alive_mask):
                batch_inputs = np.empty((self.population_size, 6), dtype=np.float64)
                for i, dino in enumerate(self.dinosaurs):
                    if dino.is_alive:
                        batch_inputs[i] = self.get_sensors_for_dino(dino)
                    else:
                        batch_inputs[i] = 0.0

                # Neural inference with normalized sensors
                norm_inputs = NeuralNetwork.normalize_inputs(batch_inputs)
                outputs = self.batch_brain.forward_batch(norm_inputs, alive_mask=alive_mask)

                for i, dino in enumerate(self.dinosaurs):
                    if dino.is_alive:
                        jump, duck, airplane = self._arbitrate_actions(
                            outputs[i, 0], outputs[i, 1], outputs[i, 2], dino.airplane_cooldown, speed_mag=speed_mag
                        )

                        if jump and dino.state != DinoState.JUMPING and dino.state != DinoState.FLYING:
                            next_obs = self.find_next_obstacle(dino.x)
                            if next_obs is None or (next_obs.x - dino.x > 350.0):
                                dino.fitness = max(0.0, dino.fitness - penalty_idle)
                            if dino is self.best_dinosaur:
                                self.sound_events.append("jump")

                        if airplane and dino.state != DinoState.FLYING and dino.airplane_cooldown <= 0:
                            if dino is self.best_dinosaur:
                                self.sound_events.append("airplane")

                        dino.apply_inputs(jump, duck, airplane, self.config.physics, speed_mag)

        elif self.mode == "versus":
            # Dino 0: Human Player
            dino_player = self.dinosaurs[0]
            if dino_player.is_alive and player_inputs is not None:
                p_jump, p_duck, p_plane = player_inputs
                if p_jump and dino_player.state != DinoState.JUMPING and dino_player.state != DinoState.FLYING:
                    self.sound_events.append("jump")
                if p_plane and dino_player.state != DinoState.FLYING and dino_player.airplane_cooldown <= 0:
                    self.sound_events.append("airplane")
                dino_player.apply_inputs(p_jump, p_duck, p_plane, self.config.physics, speed_mag)

            # Dino 1: Trained AI Opponent
            dino_ai = self.dinosaurs[1]
            if dino_ai.is_alive and self.single_brain:
                sensors = self.get_sensors_for_dino(dino_ai)
                norm_sensors = NeuralNetwork.normalize_inputs(sensors)
                outs = self.single_brain.forward(norm_sensors)
                ai_jump, ai_duck, ai_plane = self._arbitrate_actions(
                    outs[0], outs[1], outs[2], dino_ai.airplane_cooldown, speed_mag=speed_mag
                )
                dino_ai.apply_inputs(ai_jump, ai_duck, ai_plane, self.config.physics, speed_mag)

        elif self.mode == "playable":
            dino = self.dinosaurs[0]
            if dino.is_alive:
                if player_inputs is not None:
                    jump, duck, airplane = player_inputs
                    if jump and dino.state != DinoState.JUMPING and dino.state != DinoState.FLYING:
                        self.sound_events.append("jump")
                    if airplane and dino.state != DinoState.FLYING and dino.airplane_cooldown <= 0:
                        self.sound_events.append("airplane")
                    dino.apply_inputs(jump, duck, airplane, self.config.physics, speed_mag)
                elif self.single_brain:
                    sensors = self.get_sensors_for_dino(dino)
                    norm_sensors = NeuralNetwork.normalize_inputs(sensors)
                    outs = self.single_brain.forward(norm_sensors)
                    ai_jump, ai_duck, ai_plane = self._arbitrate_actions(
                        outs[0], outs[1], outs[2], dino.airplane_cooldown, speed_mag=speed_mag
                    )
                    dino.apply_inputs(ai_jump, ai_duck, ai_plane, self.config.physics, speed_mag)

        elif self.mode == "evaluation" and self.single_brain:
            dino = self.dinosaurs[0]
            if dino.is_alive:
                sensors = self.get_sensors_for_dino(dino)
                norm_sensors = NeuralNetwork.normalize_inputs(sensors)
                outs = self.single_brain.forward(norm_sensors)
                jump, duck, airplane = self._arbitrate_actions(
                    outs[0], outs[1], outs[2], dino.airplane_cooldown, speed_mag=speed_mag
                )
                if jump and dino.state != DinoState.JUMPING and dino.state != DinoState.FLYING:
                    self.sound_events.append("jump")
                if airplane and dino.state != DinoState.FLYING and dino.airplane_cooldown <= 0:
                    self.sound_events.append("airplane")
                dino.apply_inputs(jump, duck, airplane, self.config.physics, speed_mag)

        # 4. Update dinosaur physics and animations
        for dino in self.dinosaurs:
            dino.update_physics(self.config.physics, self.speed)
            dino.update_animation(dt)

        # 5. Collision checks
        for dino in self.dinosaurs:
            if dino.is_alive:
                obs = self.find_next_obstacle(dino.x)
                if obs and check_dino_obstacle_collision(
                    dino, obs,
                    h_offset=self.config.physics.hitbox_horizontal_correction,
                    v_offset=self.config.physics.hitbox_vertical_correction,
                ):
                    dino.state = DinoState.DEAD
                    self.dead_count += 1
                    if dino is self.best_dinosaur or self.mode in ("versus", "playable"):
                        self.sound_events.append("death")

        # 6. Track best alive dinosaur for camera, lasers & visualizer (foremost live runner)
        alive_dinos = [d for d in self.dinosaurs if d.is_alive]
        if alive_dinos:
            self.best_dinosaur = max(alive_dinos, key=lambda d: d.x)

        # 7. Speed acceleration & distance tracking
        old_dist = self.distance
        if speed_mag < self.config.physics.max_speed_magnitude:
            self.speed -= self.config.physics.speed_acceleration
        self.distance += speed_mag

        # Milestone sound every 1000px
        if int(self.distance) // 1000 > int(old_dist) // 1000:
            self.sound_events.append("score")

        # 8. Check match termination
        match_finished = (self.dead_count >= self.population_size) or (self.distance > 1_000_000.0)

        if match_finished:
            self._handle_match_end()
            return True

        return False

    def _handle_match_end(self) -> None:
        """Processes end of generation: GA evolution, record update, and telemetry."""
        if self.distance > self.record_distance:
            self.record_distance = self.distance

        if self.mode == "training" and self.ga and self.batch_brain:
            fitnesses = np.array([d.fitness for d in self.dinosaurs], dtype=np.float64)
            best_fit, avg_fit, champion = self.ga.evolve(fitnesses)
            std_fit = float(np.std(fitnesses))

            if self.telemetry_logger:
                self.telemetry_logger.log_generation(
                    generation=self.ga.generation,
                    best_fitness=best_fit,
                    avg_fitness=avg_fit,
                    std_fitness=std_fit,
                    record_distance=self.record_distance,
                    dead_count=self.dead_count,
                    population_size=self.population_size,
                )

            # Update batch neural network with newly mutated generation
            self.batch_brain.load_population_genomes(self.ga.population)

        self.start_new_match()
