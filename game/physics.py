"""Physics utilities including AABB collision detection with tolerance offsets."""

from typing import Tuple
from config.settings import PhysicsSettings
from game.entities.dinosaur import Dinosaur
from game.entities.obstacle import Obstacle


def aabb_intersect(
    x1: float, y1: float, w1: float, h1: float,
    x2: float, y2: float, w2: float, h2: float
) -> bool:
    """Axis-Aligned Bounding Box (AABB) intersection check."""
    if x1 + w1 <= x2:
        return False
    if x1 >= x2 + w2:
        return False
    if y1 + h1 <= y2:
        return False
    if y1 >= y2 + h2:
        return False
    return True


def check_dino_obstacle_collision(
    dino: Dinosaur,
    obstacle: Obstacle,
    h_offset: int = 7,
    v_offset: int = 5
) -> bool:
    """
    Checks collision between a dinosaur and an obstacle.
    Applies the original C tolerance offsets:
      DinoX = dino.X + h_offset
      DinoY = dino.Y + v_offset
      DinoLarg = dino.width - 2 * h_offset
      DinoAlt = dino.height - 2 * v_offset
    """
    if not dino.is_alive:
        return False

    dino_w, dino_h = dino.width_and_height
    dino_x = dino.x + h_offset
    dino_y = dino.y + v_offset
    dino_eff_w = dino_w - 2 * h_offset
    dino_eff_h = dino_h - 2 * v_offset

    obs_x = obstacle.x
    obs_y = obstacle.y
    obs_w = obstacle.width
    obs_h = obstacle.height

    return aabb_intersect(
        dino_x, dino_y, dino_eff_w, dino_eff_h,
        obs_x, obs_y, obs_w, obs_h
    )
