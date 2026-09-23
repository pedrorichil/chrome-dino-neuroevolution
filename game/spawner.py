"""Obstacle Spawner generating procedural obstacle tracks matching original patterns."""

import random
from typing import List, Tuple
from config.settings import ObstacleType
from game.entities.obstacle import Obstacle, OBSTACLE_DIMS


class ObstacleTemplate:
    """Precomputed blueprint for a single obstacle along the track."""

    def __init__(self, obs_type: ObstacleType, x: float, y: float):
        self.type = obs_type
        self.x = x
        self.y = y


class ObstacleSpawner:
    """
    Generates procedural obstacles matching the C generation logic:
    - Periodic spikes every 10 obstacles (requiring the airplane).
    - Random cactuses (types 0..4) and birds (type 5).
    - Birds at varying altitudes: Y in [20, 85].
    - Spacing: previous_x + previous_width + 500 + random(-100, 100).
    """

    def __init__(self, track_length: int = 20000, seed: int = None):
        self.track_length = track_length
        self.rng = random.Random(seed)
        self.templates: List[ObstacleTemplate] = []
        self.next_index = 0
        self.generate_track()

    def generate_track(self) -> None:
        """Builds precomputed sequence of obstacles."""
        self.templates.clear()
        self.next_index = 0

        # Initial obstacle at 1250px
        first_type = ObstacleType.BIRD
        self.templates.append(ObstacleTemplate(first_type, x=1250.0, y=15.0))

        spike_counter = 0

        for i in range(1, self.track_length):
            if spike_counter >= 10:
                obs_type = ObstacleType.SPIKES
                spike_counter = 0
            else:
                obs_type = ObstacleType(self.rng.randint(0, 5))
                spike_counter += 1

            prev = self.templates[i - 1]
            prev_width = OBSTACLE_DIMS[5 if prev.type == ObstacleType.BIRD else int(prev.type)][0]
            spacing = prev_width + 500.0 + self.rng.randint(-100, 100)
            obs_x = prev.x + spacing

            if obs_type == ObstacleType.BIRD:
                obs_y = 20.0 + self.rng.randint(0, 64)
            elif obs_type == ObstacleType.SPIKES:
                obs_y = 10.0
            else:
                obs_y = 15.0

            self.templates.append(ObstacleTemplate(obs_type, obs_x, obs_y))

    def reset(self) -> None:
        self.next_index = 0

    def spawn_initial_pool(self, max_active: int = 7) -> List[Obstacle]:
        """Spawns first N obstacles into the active simulation."""
        active: List[Obstacle] = []
        for _ in range(max_active):
            obs = self.get_next_obstacle(current_distance=0.0)
            if obs:
                active.append(obs)
        return active

    def get_next_obstacle(self, current_distance: float) -> Obstacle:
        """Returns the next obstacle adjusted for current elapsed distance."""
        if self.next_index >= len(self.templates):
            # Regenerate track if reached end of 20,000 obstacles
            self.generate_track()

        tmpl = self.templates[self.next_index]
        self.next_index += 1
        return Obstacle(tmpl.type, x=tmpl.x - current_distance, y=tmpl.y)
