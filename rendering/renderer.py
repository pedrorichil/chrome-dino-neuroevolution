"""Master Renderer orchestrating visual presentation, overlays, and audio."""

import math
from typing import Optional, Tuple, List
import pygame

from config.settings import DisplaySettings, DinoState
from game.engine import GameEngine
from rendering.assets import AssetManager
from rendering.hud import HUD
from rendering.neural_visualizer import NeuralVisualizer
from rendering.graph_visualizer import GraphVisualizer
from rendering.audio import SoundSynthesizer


class Renderer:
    """
    Renders all visual layers of the Google Dino AI simulation:
    - Clean white canvas (Classic) or Day/Night Cycle (Toggle 'N')
    - Clouds & Parallax Mountains
    - Ground Tiles
    - Obstacles
    - Dinosaurs (with alive/dead fade, leader indicator, versus labels)
    - Sensor Raycast Visualizer (Toggle 'S')
    - AABB Hitbox Debug Overlay (Toggle 'H')
    - Real-time Neural Network Architecture & Synaptic Glows
    - Historical Fitness Evolution Graph
    - Sound Synthesizer integration
    """

    def __init__(self, config: Optional[DisplaySettings] = None, assets: Optional[AssetManager] = None):
        self.config = config or DisplaySettings()
        self.screen: Optional[pygame.Surface] = None
        self.assets = assets or AssetManager(display_settings=self.config)
        self.audio = SoundSynthesizer()

        self.hud = HUD(x=20, base_y=360)
        self.neural_vis = NeuralVisualizer(x=665, y=20, width=680, height=350)
        self.graph_vis = GraphVisualizer(x=20, y=20, width=self.config.graph_width, height=320)

        # Feature toggles (Clean defaults)
        self.show_display = True
        self.show_sensors = True
        self.show_hitboxes = False
        self.enable_day_night = False

        # Stars in the open sky area (Y=380..520, below HUD/Network and above mountains)
        self.stars = [
            (int((i * 137.5) % self.config.screen_width), int((i * 43.3) % 130 + 390), (i % 2) + 1)
            for i in range(35)
        ]

    def initialize_window(self) -> None:
        """Initializes Pygame display, audio mixer, and loads assets."""
        pygame.init()
        pygame.display.set_caption(self.config.title)
        self.screen = pygame.display.set_mode(
            (self.config.screen_width, self.config.screen_height),
            pygame.DOUBLEBUF
        )
        self.assets.load_all()
        self.audio.initialize()

    def game_to_screen_y(self, game_y: float, sprite_height: int) -> int:
        """Converts bottom-up game coordinates to top-down Pygame screen coordinates."""
        return int(self.config.screen_height - game_y - sprite_height)

    def _calculate_day_night(self, distance: float) -> Tuple[Tuple[int, int, int], float]:
        """Calculates background color and darkness factor if Day/Night mode is toggled."""
        if not self.enable_day_night:
            return (255, 255, 255), 0.0

        phase = (distance % 2400.0) / 2400.0
        day_color = (255, 255, 255)
        night_color = (30, 30, 44)

        if phase < 0.45:
            return day_color, 0.0
        elif phase < 0.55:
            t = (phase - 0.45) / 0.10
            r = int(day_color[0] * (1 - t) + night_color[0] * t)
            g = int(day_color[1] * (1 - t) + night_color[1] * t)
            b = int(day_color[2] * (1 - t) + night_color[2] * t)
            return (r, g, b), t
        elif phase < 0.90:
            return night_color, 1.0
        else:
            t = (phase - 0.90) / 0.10
            r = int(night_color[0] * (1 - t) + day_color[0] * t)
            g = int(night_color[1] * (1 - t) + day_color[1] * t)
            b = int(night_color[2] * (1 - t) + day_color[2] * t)
            return (r, g, b), 1.0 - t

    def render(self, engine: GameEngine, clock_period: float = 0.005, sim_speed: int = 1) -> None:
        """Renders one full frame."""
        # Process audio events triggered by engine
        for ev in engine.sound_events:
            if ev == "jump":
                self.audio.play_jump()
            elif ev == "airplane":
                self.audio.play_airplane()
            elif ev == "death":
                self.audio.play_death()
            elif ev == "score":
                self.audio.play_score()

        if not self.show_display or self.screen is None:
            return

        # 1. Background color
        bg_color, darkness = self._calculate_day_night(engine.distance)
        is_dark = darkness > 0.4
        self.screen.fill(bg_color)

        # Render Moon and Stars in game sky (avoiding dashboard)
        if darkness > 0.3:
            alpha = int(darkness * 220)
            for sx, sy, s_rad in self.stars:
                star_surf = pygame.Surface((s_rad * 2, s_rad * 2), pygame.SRCALPHA)
                pygame.draw.circle(star_surf, (240, 240, 255, alpha), (s_rad, s_rad), s_rad)
                self.screen.blit(star_surf, (sx, sy))

            # Crescent Moon in the sky
            moon_x, moon_y = 520, 420
            moon_surf = pygame.Surface((44, 44), pygame.SRCALPHA)
            pygame.draw.circle(moon_surf, (245, 245, 220, alpha), (22, 22), 16)
            pygame.draw.circle(moon_surf, (bg_color[0], bg_color[1], bg_color[2], alpha), (28, 18), 13)
            self.screen.blit(moon_surf, (moon_x, moon_y))

        # 2. Clouds
        if self.assets.cloud_sprite:
            for cloud in engine.environment.clouds:
                sy = self.game_to_screen_y(cloud.y, cloud.height)
                self.screen.blit(self.assets.cloud_sprite, (int(cloud.x), sy))

        # 3. Parallax Mountains
        for layer in engine.environment.mountains:
            for j in range(2):
                code = layer.layer_idx * 10 + (j + 1)
                m_surf = self.assets.mountain_sprites.get(code)
                if m_surf:
                    sy = self.game_to_screen_y(layer.y[j], layer.height)
                    self.screen.blit(m_surf, (int(layer.x[j]), sy))

        # 4. Ground
        for tile in engine.environment.ground_tiles:
            if tile.variant < len(self.assets.ground_sprites):
                g_surf = self.assets.ground_sprites[tile.variant]
                sy = self.game_to_screen_y(tile.y, tile.height)
                self.screen.blit(g_surf, (int(tile.x), sy))

        # 5. Obstacles
        for obs in engine.obstacles:
            if obs.x + obs.width < -20 or obs.x > self.config.screen_width + 20:
                continue
            s_idx = obs.sprite_index
            if s_idx < len(self.assets.obstacle_sprites):
                obs_surf = self.assets.obstacle_sprites[s_idx]
                sy = self.game_to_screen_y(obs.y, obs.height)
                self.screen.blit(obs_surf, (int(obs.x), sy))

                # Hitbox overlay for obstacle
                if self.show_hitboxes:
                    pygame.draw.rect(
                        self.screen, (255, 165, 0),
                        pygame.Rect(int(obs.x), sy, obs.width, obs.height), 2
                    )

        # 6. Dinosaurs
        sorted_dinos = sorted(engine.dinosaurs, key=lambda d: 1 if d.is_alive else 0)
        champion_dino = engine.best_dinosaur

        for dino in sorted_dinos:
            d_w, d_h = dino.width_and_height
            if dino.x + d_w < -20 or dino.x > self.config.screen_width + 20:
                continue

            d_surf = self.assets.dino_sprites.get((dino.color_idx, dino.sprite_index))
            sy = self.game_to_screen_y(dino.y, d_h)

            if d_surf:
                if not dino.is_alive:
                    # Smooth fade out as dead dinosaur slides back
                    fade = max(0.0, min(1.0, (dino.x - 30.0) / 180.0))
                    if fade <= 0.05:
                        continue
                    dead_surf = d_surf.copy()
                    dead_surf.set_alpha(int(fade * 130))
                    self.screen.blit(dead_surf, (int(dino.x), sy))
                else:
                    self.screen.blit(d_surf, (int(dino.x), sy))

            # Airplane sprite if flying
            if dino.state == DinoState.FLYING and len(self.assets.airplane_sprites) > dino.airplane_anim_frame:
                plane_surf = self.assets.airplane_sprites[dino.airplane_anim_frame]
                plane_sy = self.game_to_screen_y(dino.y + 12.0, 37)
                self.screen.blit(plane_surf, (int(dino.x + 13.0), plane_sy))

            # Hitbox overlay for dinosaur
            if self.show_hitboxes and dino.is_alive:
                h_off = engine.config.physics.hitbox_horizontal_correction
                v_off = engine.config.physics.hitbox_vertical_correction
                box_x = int(dino.x + h_off)
                box_y = int(sy + v_off)
                box_w = d_w - 2 * h_off
                box_h = d_h - 2 * v_off
                pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(box_x, box_y, box_w, box_h), 2)

            # Highlight foremost leader dinosaur with an arrow indicator
            if dino is champion_dino and dino.is_alive and engine.mode != "versus":
                marker_x = int(dino.x + d_w / 2.0)
                marker_y = sy - 8
                pygame.draw.polygon(
                    self.screen,
                    (255, 215, 0),
                    [(marker_x, marker_y), (marker_x - 5, marker_y - 8), (marker_x + 5, marker_y - 8)]
                )

            # Versus Mode Labels
            if engine.mode == "versus":
                font_v = self.assets.font_small or pygame.font.SysFont("Arial", 12)
                v_label = "VOCÊ" if dino.index == 0 else "I.A. CAMPEÃ"
                v_color = (0, 180, 0) if dino.index == 0 else (30, 144, 255)
                lbl_surf = font_v.render(v_label, True, v_color)
                self.screen.blit(lbl_surf, (int(dino.x - 5), sy - 18))

        # 7. Sensor Raycast (Targeting Laser) from champion leader to obstacle
        closest_obs = engine.find_next_obstacle(champion_dino.x)
        if self.show_sensors and champion_dino.is_alive and closest_obs:
            c_w, c_h = champion_dino.width_and_height
            eye_x = int(champion_dino.x + c_w - 4)
            eye_y = self.game_to_screen_y(champion_dino.y, c_h) + 12

            target_x = int(closest_obs.x)
            target_y = self.game_to_screen_y(closest_obs.y, closest_obs.height) + closest_obs.height // 2

            # Laser line
            pygame.draw.line(self.screen, (220, 20, 60), (eye_x, eye_y), (target_x, target_y), 2)
            pygame.draw.circle(self.screen, (220, 20, 60), (target_x, target_y), 4)

            # Distance pill tag
            font_s = self.assets.font_small or pygame.font.SysFont("Arial", 11)
            dist_val = max(0.0, closest_obs.x - champion_dino.x)
            dist_txt = font_s.render(f"{dist_val:.0f}px", True, (220, 20, 60))
            mid_x = (eye_x + target_x) // 2
            mid_y = (eye_y + target_y) // 2 - 14

            # Background pill for readability
            pill_rect = pygame.Rect(mid_x - 3, mid_y - 2, dist_txt.get_width() + 6, dist_txt.get_height() + 4)
            pygame.draw.rect(self.screen, (255, 255, 255, 220), pill_rect, border_radius=4)
            self.screen.blit(dist_txt, (mid_x, mid_y))

        # 8. Real-time Neural Network Architecture
        sensors = engine.get_sensors_for_dino(champion_dino)
        if engine.mode == "training" and engine.ga and champion_dino.index < len(engine.ga.population):
            single_brain = getattr(self, "_cached_vis_brain", None)
            if single_brain is None:
                from core.neural_network import NeuralNetwork
                single_brain = NeuralNetwork(engine.config.network)
                self._cached_vis_brain = single_brain

            single_brain.load_genome(engine.ga.population[champion_dino.index])
            single_brain.forward(sensors)
            self.neural_vis.draw(self.screen, single_brain, self.assets, sensors, dark_mode=is_dark)
        elif engine.single_brain:
            engine.single_brain.forward(sensors)
            self.neural_vis.draw(self.screen, engine.single_brain, self.assets, sensors, dark_mode=is_dark)

        # 9. Historical Fitness Graph
        best_hist = engine.ga.best_fitness_history if engine.ga else []
        avg_hist = engine.ga.avg_fitness_history if engine.ga else []
        cur_best = champion_dino.fitness
        alive_dinos = [d for d in engine.dinosaurs if d.is_alive]
        cur_avg = sum(d.fitness for d in engine.dinosaurs) / max(1, len(engine.dinosaurs))

        self.graph_vis.draw(
            self.screen, best_hist, avg_hist, self.assets,
            current_best=cur_best, current_avg=cur_avg, dark_mode=is_dark
        )

        # 10. HUD
        gen = engine.ga.generation if engine.ga else 1
        alive_count = len(alive_dinos)
        self.hud.draw(
            surface=self.screen,
            assets=self.assets,
            generation=gen,
            clock_period=clock_period,
            speed=engine.speed,
            record_distance=engine.record_distance,
            current_distance=engine.distance,
            alive_count=alive_count,
            total_population=engine.population_size,
            dark_mode=is_dark,
            sim_speed=sim_speed,
        )

        # Hotkey status hints at top right
        hint_font = self.assets.font_small or pygame.font.SysFont("Arial", 11)
        mute_txt = "[M] Mudo: SIM" if self.audio.muted else "[M] Mudo: NÃO"
        sens_txt = "[S] Raios: ON" if self.show_sensors else "[S] Raios: OFF"
        hit_txt = "[H] Hitbox: ON" if self.show_hitboxes else "[H] Hitbox: OFF"
        night_txt = "[N] Noite: ON" if self.enable_day_night else "[N] Noite: OFF"
        speed_txt = f"[+/-] {sim_speed}x"
        hints = f"{speed_txt}  |  {sens_txt}  |  {hit_txt}  |  {night_txt}  |  {mute_txt}  |  [ESC] Turbo"
        hint_color = (200, 200, 220) if is_dark else (100, 100, 100)
        hint_surf = hint_font.render(hints, True, hint_color)
        self.screen.blit(hint_surf, (self.config.screen_width - 500, 10))

        pygame.display.flip()
