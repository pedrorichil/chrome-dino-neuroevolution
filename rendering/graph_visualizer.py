"""Real-time fitness evolution graph displaying Best and Average metrics per generation."""

from typing import List
import pygame
from rendering.assets import AssetManager


class GraphVisualizer:
    """
    Renders an evolutionary progress chart:
    - Blue line: Champion fitness per generation.
    - Red line: Population mean fitness per generation.
    - Coordinate grid and real-time legends.
    """

    def __init__(self, x: int = 20, y: int = 390, width: int = 600, height: int = 350):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(
        self,
        surface: pygame.Surface,
        best_history: List[float],
        avg_history: List[float],
        assets: AssetManager,
        current_best: float = 0.0,
        current_avg: float = 0.0,
        dark_mode: bool = False,
    ) -> None:
        """Draws the grid, historical data points, and connection lines."""
        # 1. Background grid (10 divisions)
        grid_color = (55, 55, 75) if dark_mode else (230, 230, 230)
        axis_color = (180, 180, 200) if dark_mode else (100, 100, 100)

        for i in range(1, 11):
            gx = self.x + i * (self.width // 10)
            gy = self.y + i * (self.height // 10)
            pygame.draw.line(surface, grid_color, (gx, self.y), (gx, self.y + self.height), 1)
            pygame.draw.line(surface, grid_color, (self.x, gy), (self.x + self.width, gy), 1)

        # Coordinate axes
        pygame.draw.line(surface, axis_color, (self.x, self.y), (self.x, self.y + self.height), 2)
        pygame.draw.line(surface, axis_color, (self.x, self.y + self.height), (self.x + self.width, self.y + self.height), 2)

        # 2. History plotting based on generation index
        history_len = len(best_history)
        if history_len == 0 and current_best <= 0:
            return

        # Always scale X to at least 10 generations, growing as evolution progresses
        total_slots = max(10, history_len + 1)
        scale_x = self.width / total_slots

        # Find global peak fitness for Y scaling
        all_vals = best_history + avg_history + [current_best, current_avg]
        max_fitness = max(100.0, max(all_vals) if all_vals else 100.0)
        scale_y = (self.height - 30) / max_fitness

        best_points = []
        avg_points = []

        # Completed generations
        for i in range(history_len):
            px = int(self.x + (i + 1) * scale_x)
            by = int(self.y + self.height - (best_history[i] * scale_y))
            ay = int(self.y + self.height - (avg_history[i] * scale_y))
            best_points.append((px, by))
            avg_points.append((px, ay))

        # Current ongoing generation (plotted at next slot)
        if current_best > 0:
            cur_px = int(self.x + (history_len + 1) * scale_x)
            cur_by = int(self.y + self.height - (current_best * scale_y))
            cur_ay = int(self.y + self.height - (current_avg * scale_y))
            best_points.append((cur_px, cur_by))
            avg_points.append((cur_px, cur_ay))

        # Draw lines
        if len(best_points) >= 2:
            pygame.draw.lines(surface, (30, 144, 255), False, best_points, 2)  # Blue
            for pt in best_points:
                pygame.draw.circle(surface, (30, 144, 255), pt, 3)
        elif len(best_points) == 1:
            pygame.draw.circle(surface, (30, 144, 255), best_points[0], 4)

        if len(avg_points) >= 2:
            pygame.draw.lines(surface, (220, 20, 60), False, avg_points, 2)   # Red
            for pt in avg_points:
                pygame.draw.circle(surface, (220, 20, 60), pt, 3)
        elif len(avg_points) == 1:
            pygame.draw.circle(surface, (220, 20, 60), avg_points[0], 4)

        # Legends
        font = assets.font_small or pygame.font.SysFont("Arial", 12)
        b_color = (50, 160, 255) if dark_mode else (30, 144, 255)
        a_color = (255, 80, 80) if dark_mode else (220, 20, 60)
        legend_best = font.render(f"Melhor: {max_fitness:.0f}", True, b_color)
        legend_avg = font.render("Média Populacional", True, a_color)
        surface.blit(legend_best, (self.x + 10, self.y + 10))
        surface.blit(legend_avg, (self.x + 130, self.y + 10))
