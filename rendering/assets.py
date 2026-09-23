"""Asset manager for loading, colorkeying, tinting, and caching game sprites and fonts."""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pygame

from config.settings import DisplaySettings


class AssetManager:
    """Manages sprite sheets, color palettes, and fonts for the game."""

    def __init__(self, assets_dir: Optional[Path] = None, display_settings: Optional[DisplaySettings] = None):
        self.assets_dir = assets_dir or Path(__file__).resolve().parent.parent / "assets"
        self.display = display_settings or DisplaySettings()

        self.img_dir = self.assets_dir / "imagens"
        self.font_dir = self.assets_dir / "fontes"

        # Cached surfaces
        self.dino_sprites: Dict[Tuple[int, int], pygame.Surface] = {}  # (color_idx, frame_idx) -> Surface
        self.airplane_sprites: List[pygame.Surface] = []
        self.obstacle_sprites: List[pygame.Surface] = []
        self.ground_sprites: List[pygame.Surface] = []
        self.mountain_sprites: Dict[int, pygame.Surface] = {}
        self.cloud_sprite: Optional[pygame.Surface] = None

        # Visualizer sprites
        self.neuron_sprite: Optional[pygame.Surface] = None
        self.light_sprite: Optional[pygame.Surface] = None
        self.arrow_sprite: Optional[pygame.Surface] = None

        # Fonts
        self.font_small: Optional[pygame.font.Font] = None
        self.font_medium: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None

        self._loaded = False

    def load_all(self) -> None:
        """Loads and pre-processes all visual assets."""
        if self._loaded:
            return

        self._load_fonts()
        self._load_dinosaurs()
        self._load_airplanes()
        self._load_obstacles()
        self._load_environment()
        self._load_visualizer_assets()

        self._loaded = True

    def _load_image(self, filename: str, is_bmp: bool = True) -> pygame.Surface:
        """Loads an image with transparent colorkey or alpha channel."""
        filepath = self.img_dir / filename
        if not filepath.exists():
            # Fallback placeholder if asset is missing
            surf = pygame.Surface((40, 40), pygame.SRCALPHA)
            surf.fill((255, 0, 255, 200))
            return surf

        surf = pygame.image.load(str(filepath))
        if is_bmp or filename.lower().endswith(".bmp"):
            colorkey = surf.get_at((0, 0))
            surf.set_colorkey(colorkey)
            surf = surf.convert_alpha()
        else:
            surf = surf.convert_alpha()

        return surf

    def _load_fonts(self) -> None:
        """Loads TTF fonts with fallbacks."""
        ttf_path = self.font_dir / "arial.ttf"
        font_source = str(ttf_path) if ttf_path.exists() else None

        self.font_small = pygame.font.Font(font_source, 14)
        self.font_medium = pygame.font.Font(font_source, 16)
        self.font_large = pygame.font.Font(font_source, 22)

    def _load_dinosaurs(self) -> None:
        """Pre-renders 10 frames for all 8 dinosaur color variants."""
        base_dino_frames = [self._load_image(f"dino{i}.bmp") for i in range(10)]

        for color_idx, rgb in enumerate(self.display.colors):
            rgba = rgb + (255,)
            for frame_idx, base_surf in enumerate(base_dino_frames):
                tinted = base_surf.copy()
                tinted.fill(rgba, special_flags=pygame.BLEND_RGBA_MULT)
                self.dino_sprites[(color_idx, frame_idx)] = tinted

    def _load_airplanes(self) -> None:
        """Loads the 2 airplane animation frames."""
        self.airplane_sprites = [
            self._load_image(f"aviao{i}.bmp") for i in range(2)
        ]

    def _load_obstacles(self) -> None:
        """Loads obstacles 0..6 (BMPs) and 7 (PNG Spikes)."""
        self.obstacle_sprites = [
            self._load_image(f"obs{i}.bmp") for i in range(7)
        ]
        self.obstacle_sprites.append(self._load_image("obs7.png", is_bmp=False))

    def _load_environment(self) -> None:
        """Loads ground tiles, mountain backgrounds, and cloud."""
        self.ground_sprites = [
            self._load_image(f"chao{i}.bmp") for i in range(6)
        ]

        # Mountains: fundo0.bmp .. fundo5.bmp
        for code, fname in [
            (1, "fundo0.bmp"), (2, "fundo1.bmp"),
            (11, "fundo2.bmp"), (12, "fundo3.bmp"),
            (21, "fundo4.bmp"), (22, "fundo5.bmp")
        ]:
            self.mountain_sprites[code] = self._load_image(fname)

        self.cloud_sprite = self._load_image("nuvem.bmp")

    def _load_visualizer_assets(self) -> None:
        """Loads neural network visualizer elements."""
        self.neuron_sprite = self._load_image("neuronio7.png", is_bmp=False)
        self.light_sprite = self._load_image("luz.png", is_bmp=False)
        raw_arrow = self._load_image("seta2.png", is_bmp=False)
        scaled_arrow = pygame.transform.smoothscale(raw_arrow, (36, 12))
        tinted_arrow = scaled_arrow.copy()
        tinted_arrow.fill((60, 60, 60, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.arrow_sprite = tinted_arrow
