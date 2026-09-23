"""Dinosaur entity representing state, animation, physics attributes, and fitness."""

from typing import Tuple
from config.settings import DinoState, PhysicsSettings


# Dimensions for each of the 10 dino frames (from original Sprites.h)
DINO_FRAME_DIMS = [
    (40, 43), (40, 43),  # Running (frames 0, 1)
    (55, 25), (55, 26),  # Ducking (frames 2, 3)
    (40, 43), (40, 43),  # Jumping (frames 4, 5)
    (40, 43), (40, 43),  # Dead    (frames 6, 7)
    (39, 37), (39, 37),  # Flying  (frames 8, 9)
]

AIRPLANE_DIMS = (70, 37)


class Dinosaur:
    """Represents a single dinosaur agent in the simulation."""

    def __init__(self, index: int, x: float = 300.0, y: float = 15.0, color_idx: int = 0):
        self.index = index
        self.x = float(x)
        self.y = float(y)
        self.velocity_y = 0.0
        self.state = DinoState.RUNNING
        self.color_idx = color_idx % 8

        # Animation states
        self.anim_frame = 0
        self.airplane_anim_frame = 0
        self.anim_timer = 0.0
        self.airplane_anim_timer = 0.0

        # Special action states
        self.airplane_distance = 0.0
        self.airplane_cooldown = 0.0

        # Performance metric
        self.fitness = 0.0

    @property
    def is_alive(self) -> bool:
        return self.state != DinoState.DEAD

    @property
    def sprite_index(self) -> int:
        """Calculates current sprite index (0..9) based on state and animation frame."""
        base_offsets = {
            DinoState.RUNNING: 0,
            DinoState.DUCKING: 2,
            DinoState.JUMPING: 4,
            DinoState.DEAD: 6,
            DinoState.FLYING: 8,
        }
        return base_offsets[self.state] + self.anim_frame

    @property
    def width_and_height(self) -> Tuple[int, int]:
        """Returns (width, height) of current bounding box."""
        return DINO_FRAME_DIMS[self.sprite_index]

    def reset(self, x: float, y: float = 15.0) -> None:
        """Resets the dinosaur for a new match."""
        self.x = float(x)
        self.y = float(y)
        self.velocity_y = 0.0
        self.state = DinoState.RUNNING
        self.anim_frame = 0
        self.airplane_anim_frame = 0
        self.anim_timer = 0.0
        self.airplane_anim_timer = 0.0
        self.airplane_distance = 0.0
        self.airplane_cooldown = 0.0
        self.fitness = 0.0

    def apply_inputs(self, jump: bool, duck: bool, airplane: bool, physics: PhysicsSettings, speed_magnitude: float) -> None:
        """Applies neural or human control decisions according to original C game rules."""
        if not self.is_alive:
            return

        if self.state != DinoState.FLYING:
            # Revert from ducking to running if duck is not held and not jumping
            if self.state != DinoState.JUMPING:
                self.state = DinoState.RUNNING

            # Ducking action
            if duck:
                if self.state != DinoState.JUMPING:
                    self.state = DinoState.DUCKING
                else:
                    # In mid-air: downward fast-fall
                    if self.velocity_y > 0:
                        self.velocity_y = 0.0
                    self.y -= physics.duck_fall_speed

            # Jumping action
            if jump and self.state != DinoState.JUMPING:
                self.state = DinoState.JUMPING
                self.y += 1.0
                self.velocity_y += physics.jump_impulse

            # Airplane action
            if airplane and self.airplane_cooldown <= 0.0:
                self.state = DinoState.FLYING
                self.y += 1.0
                if self.velocity_y <= 0.5 and self.y < 25.0:
                    self.velocity_y += physics.jump_impulse
                self.airplane_cooldown = physics.airplane_cooldown_initial
                self.airplane_distance = 0.0
        else:
            # Already flying
            if self.airplane_distance >= physics.airplane_distance_max:
                self.airplane_distance = 0.0
                self.state = DinoState.JUMPING
            else:
                self.airplane_distance += speed_magnitude

        self.airplane_cooldown -= speed_magnitude

    def update_physics(self, physics: PhysicsSettings, speed: float) -> None:
        """Updates dinosaur vertical gravity and horizontal motion when dead."""
        if self.state == DinoState.DEAD:
            self.x += speed
            # Apply gravity to falling dead dinosaur until it hits ground
            if self.y > physics.ground_y:
                self.velocity_y -= physics.gravity
                self.y += self.velocity_y
                if self.y < physics.ground_y:
                    self.y = physics.ground_y
                    self.velocity_y = 0.0
            return

        # Gravity logic for alive dinosaurs
        if self.y > physics.ground_y:
            if self.state != DinoState.FLYING:
                self.velocity_y -= physics.gravity
            else:
                # Gliding during airplane flight
                if self.velocity_y <= 0.0:
                    self.velocity_y = 0.0
                else:
                    self.velocity_y -= physics.gravity

            self.y += self.velocity_y
        else:
            self.velocity_y = 0.0
            self.y = physics.ground_y
            if self.state == DinoState.JUMPING:
                self.state = DinoState.RUNNING

        # Fitness accrual
        if self.state in (DinoState.RUNNING, DinoState.DUCKING):
            self.fitness += 2.0
        else:
            self.fitness += 1.0

    def update_animation(self, dt: float = 0.005) -> None:
        """Steps frame animation timers."""
        self.anim_timer += dt
        if self.anim_timer >= 0.1:
            self.anim_frame = (self.anim_frame + 1) % 2
            self.anim_timer = 0.0

        if self.state == DinoState.FLYING:
            self.airplane_anim_timer += dt
            if self.airplane_anim_timer >= 0.03:
                self.airplane_anim_frame = (self.airplane_anim_frame + 1) % 2
                self.airplane_anim_timer = 0.0
