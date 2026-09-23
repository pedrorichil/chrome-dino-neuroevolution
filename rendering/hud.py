"""Heads-Up Display (HUD) presenting simulation statistics and distance counters."""

import pygame
from rendering.assets import AssetManager


class HUD:
    """Renders textual simulation statistics matching the original C interface."""

    def __init__(self, x: int = 20, base_y: int = 340):
        self.x = x
        self.base_y = base_y

    def draw(
        self,
        surface: pygame.Surface,
        assets: AssetManager,
        generation: int,
        clock_period: float,
        speed: float,
        record_distance: float,
        current_distance: float,
        alive_count: int,
        total_population: int,
        dark_mode: bool = False,
        sim_speed: int = 1,
    ) -> None:
        font = assets.font_medium or pygame.font.SysFont("Arial", 15)
        font_large = assets.font_large or pygame.font.SysFont("Arial", 18)

        text_color = (240, 240, 245) if dark_mode else (30, 30, 30)
        blue = (50, 160, 255) if dark_mode else (30, 144, 255)
        green = (50, 205, 50) if dark_mode else (34, 139, 34)

        speed_mag = abs(speed)
        px_per_sec = speed_mag / max(1e-4, clock_period) * sim_speed

        speed_label = f"Velocidade: {speed_mag:.2f} ({px_per_sec:.0f} px/s) [{sim_speed}x]" if sim_speed > 1 else f"Velocidade: {speed_mag:.2f} ({px_per_sec:.0f} px/s)"

        lines = [
            (f"Geração: {generation}", text_color, font_large),
            (f"Vivos: {alive_count} / {total_population}", green, font),
            (f"Clock: {clock_period:.4f} segundo", text_color, font),
            (speed_label, text_color, font),
        ]

        cur_y = self.base_y
        for text, color, f_obj in lines:
            rendered = f_obj.render(text, True, color)
            surface.blit(rendered, (self.x, cur_y))
            cur_y += 24

        # Record Distance
        rec_label = font.render("Distancia Recorde:", True, text_color)
        rec_val = font.render(f"{record_distance:.0f} pixels", True, blue)
        surface.blit(rec_label, (self.x, cur_y))
        surface.blit(rec_val, (self.x + 150, cur_y))
        cur_y += 22

        # Current Distance
        cur_label = font.render("Distancia Atual:", True, text_color)
        cur_val = font.render(f"{current_distance:.0f} pixels", True, text_color)
        surface.blit(cur_label, (self.x, cur_y))
        surface.blit(cur_val, (self.x + 150, cur_y))
