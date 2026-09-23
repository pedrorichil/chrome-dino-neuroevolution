"""Obstacle entities (Cacti, Birds, and Spikes) with animations and hitbox dimensions."""

from typing import Tuple
from config.settings import ObstacleType

# Bounding box dimensions for obstacle sprites 0..7
OBSTACLE_DIMS = [
    (32, 33),   # 0: Cactus Small 1
    (23, 46),   # 1: Cactus Small 2
    (15, 33),   # 2: Cactus Small 3
    (49, 33),   # 3: Cactus Large 1
    (73, 47),   # 4: Cactus Large 2
    (42, 36),   # 5: Bird Frame 0
    (42, 36),   # 6: Bird Frame 1
    (810, 31),  # 7: Spikes (obs7.png)
]


class Obstacle:
    """Represents an active obstacle moving across the screen."""

    def __init__(self, obs_type: ObstacleType, x: float, y: float):
        self.type = obs_type
        self.x = float(x)
        self.y = float(y)
        self.frame = 0
        self.timer_frame = 0.0

    @property
    def sprite_index(self) -> int:
        """Maps obstacle type and animation frame to sprite sheet index."""
        if self.type == ObstacleType.BIRD:
            return 5 if self.frame == 0 else 6
        elif self.type == ObstacleType.SPIKES:
            return 7
        else:
            return int(self.type)

    @property
    def width(self) -> int:
        return OBSTACLE_DIMS[self.sprite_index][0]

    @property
    def height(self) -> int:
        return OBSTACLE_DIMS[self.sprite_index][1]

    def update(self, speed: float, dt: float = 0.005) -> None:
        """Moves obstacle and steps animation frames for flying birds."""
        self.x += speed

        if self.type == ObstacleType.BIRD:
            self.timer_frame += dt
            if self.timer_frame >= 0.2:
                self.frame = (self.frame + 1) % 2
                self.timer_frame = 0.0
