"""Environment entities: Multi-layer Parallax Mountains, Clouds, and Infinite Ground."""

import random
from typing import List, Tuple
from config.settings import DisplaySettings


class GroundSegment:
    """Single tile of infinite scrolling ground."""

    def __init__(self, x: float, y: float = 25.0, sprite_variant: int = 0):
        self.x = float(x)
        self.y = float(y)
        self.width = 60
        self.height = 12
        self.variant = sprite_variant  # chao0.bmp .. chao5.bmp


class Cloud:
    """Individual cloud moving across the sky."""

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.width = 46
        self.height = 13


class MountainLayer:
    """Two-tile seamless parallax mountain layer."""

    def __init__(self, layer_idx: int, screen_width: int):
        self.layer_idx = layer_idx
        self.screen_width = screen_width
        self.x = [screen_width / 2.0, screen_width / 2.0 + screen_width]
        self.y = [90.0, 90.0]
        self.width = 1366
        self.height = 180


class Environment:
    """Coordinates all decorative background parallax elements."""

    def __init__(self, display_settings: DisplaySettings):
        self.display = display_settings

        # 30 Ground tiles
        self.ground_tiles: List[GroundSegment] = []
        variant_counter = 0
        for i in range(30):
            if variant_counter == 4:
                variant = random.randint(4, 5)
                variant_counter = 0
            else:
                variant = random.randint(0, 3)
                variant_counter += 1
            self.ground_tiles.append(GroundSegment(x=i * 60.0, y=8.0, sprite_variant=variant))

        # 15 Clouds with non-overlapping heuristic
        self.clouds: List[Cloud] = []
        for _ in range(15):
            for _ in range(50):
                cx = float(random.randint(50, 1150))
                cy = float(random.randint(100, 240))
                # Check minimum spacing
                if all((abs(cx - c.x) ** 2 + abs(cy - c.y) ** 2) ** 0.5 >= 46.0 for c in self.clouds):
                    self.clouds.append(Cloud(cx, cy))
                    break
            else:
                self.clouds.append(Cloud(random.uniform(50, 1150), random.uniform(100, 240)))

        # 3 Mountain layers
        self.mountains: List[MountainLayer] = [
            MountainLayer(layer_idx=i, screen_width=self.display.screen_width)
            for i in range(3)
        ]

    def update(self, speed: float) -> None:
        """Updates positions of ground, clouds, and mountains based on game speed."""
        # Update ground
        total_ground_span = len(self.ground_tiles) * 60.0
        for tile in self.ground_tiles:
            tile.x += speed
            if tile.x + tile.width / 2.0 < 0:
                tile.x += total_ground_span

        # Update clouds
        for cloud in self.clouds:
            cloud.x += speed * 0.01
            if cloud.x < -23.0:
                cloud.x += self.display.screen_width + 46.0

        # Update parallax mountains
        for i, layer in enumerate(self.mountains):
            layer_factor = (i * 0.02) + 0.01
            for j in range(2):
                layer.x[j] += speed * layer_factor
                if layer.x[j] < -self.display.screen_width / 2.0:
                    layer.x[j] += 2.0 * self.display.screen_width
