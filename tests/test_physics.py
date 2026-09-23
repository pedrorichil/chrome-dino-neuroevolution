"""Unit tests for Dinosaur physics, collision detection, and mechanics."""

import pytest
from config.settings import PhysicsSettings, DinoState, ObstacleType
from game.entities.dinosaur import Dinosaur
from game.entities.obstacle import Obstacle
from game.physics import aabb_intersect, check_dino_obstacle_collision


def test_aabb_intersection():
    # Box 1: (0, 0, 10, 10), Box 2: (5, 5, 10, 10) -> Intersects
    assert aabb_intersect(0, 0, 10, 10, 5, 5, 10, 10) is True

    # Box 1: (0, 0, 10, 10), Box 2: (11, 0, 10, 10) -> Apart horizontally
    assert aabb_intersect(0, 0, 10, 10, 11, 0, 10, 10) is False

    # Box 1: (0, 0, 10, 10), Box 2: (0, 11, 10, 10) -> Apart vertically
    assert aabb_intersect(0, 0, 10, 10, 0, 11, 10, 10) is False


def test_dinosaur_jump_and_gravity():
    physics = PhysicsSettings()
    dino = Dinosaur(index=0, x=300.0, y=15.0)

    # Initial state
    assert dino.state == DinoState.RUNNING
    assert dino.y == 15.0

    # Apply jump
    dino.apply_inputs(jump=True, duck=False, airplane=False, physics=physics, speed_magnitude=3.0)
    assert dino.state == DinoState.JUMPING
    assert dino.y == 16.0
    assert dino.velocity_y == 4.0

    # Step physics (gravity reduces velocity)
    dino.update_physics(physics=physics, speed=-3.0)
    assert dino.velocity_y == pytest.approx(4.0 - 0.08)
    assert dino.y > 16.0


def test_dinosaur_duck_fall():
    physics = PhysicsSettings()
    dino = Dinosaur(index=0, x=300.0, y=50.0)
    dino.state = DinoState.JUMPING
    dino.velocity_y = 2.0

    # Apply duck while in mid-air
    dino.apply_inputs(jump=False, duck=True, airplane=False, physics=physics, speed_magnitude=3.0)
    # Fast fall cancels positive velocity and reduces Y
    assert dino.velocity_y == 0.0
    assert dino.y == 48.0


def test_collision_with_hitbox_correction():
    dino = Dinosaur(index=0, x=300.0, y=15.0)
    # Obstacle directly overlapping dinosaur
    obs = Obstacle(obs_type=ObstacleType.CACTUS_SMALL_1, x=310.0, y=15.0)

    assert check_dino_obstacle_collision(dino, obs) is True

    # Obstacle far ahead
    obs_far = Obstacle(obs_type=ObstacleType.CACTUS_SMALL_1, x=600.0, y=15.0)
    assert check_dino_obstacle_collision(dino, obs_far) is False


def test_dinosaur_mass_and_weight_physics():
    physics = PhysicsSettings(gravity=0.08, jump_impulse=4.0, mass=1.0)
    dino_light = Dinosaur(index=0, x=300.0, y=15.0)
    dino_light.mass = 0.5

    dino_heavy = Dinosaur(index=1, x=300.0, y=15.0)
    dino_heavy.mass = 2.0

    # Ambos pulam: dino mais leve ganha maior velocidade vertical
    dino_light.apply_inputs(jump=True, duck=False, airplane=False, physics=physics, speed_magnitude=3.0)
    dino_heavy.apply_inputs(jump=True, duck=False, airplane=False, physics=physics, speed_magnitude=3.0)

    assert dino_light.velocity_y > dino_heavy.velocity_y
    assert dino_light.velocity_y == pytest.approx(4.0 / 0.5)
    assert dino_heavy.velocity_y == pytest.approx(4.0 / 2.0)

    # Gravidade proporcional a massa: o mais pesado desacelera mais rápido
    dino_light.update_physics(physics=physics, speed=-3.0)
    dino_heavy.update_physics(physics=physics, speed=-3.0)

    # Variação de velocidade para o pesado é maior
    expected_grav_light = 0.08 * (0.5 / 1.0)
    expected_grav_heavy = 0.08 * (2.0 / 1.0)
    assert dino_light.velocity_y == pytest.approx((4.0 / 0.5) - expected_grav_light)
    assert dino_heavy.velocity_y == pytest.approx((4.0 / 2.0) - expected_grav_heavy)
