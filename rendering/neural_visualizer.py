"""Real-time neural network graph visualizer displaying activations and weights."""

from typing import Tuple, List, Optional
import numpy as np
import pygame

from rendering.assets import AssetManager
from core.neural_network import NeuralNetwork


class NeuralVisualizer:
    """
    Renders an animated architectural diagram of the champion dinosaur's neural network,
    including real-time sensor readouts, synaptic weight lines, and glowing active neurons.
    Adapts gracefully to any number of hidden neurons.
    """

    def __init__(self, x: int = 665, y: int = 360, width: int = 700, height: int = 350):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(
        self,
        surface: pygame.Surface,
        brain: NeuralNetwork,
        assets: AssetManager,
        sensor_values: np.ndarray,
        dark_mode: bool = False,
    ) -> None:
        """Draws the neural topology and real-time signals."""
        num_inputs = brain.settings.num_inputs
        num_hidden = brain.settings.num_hidden_neurons
        num_outputs = brain.settings.num_outputs

        text_color = (235, 235, 245) if dark_mode else (40, 40, 40)
        inactive_out_color = (160, 160, 170) if dark_mode else (80, 80, 80)

        # Origin coordinates (bottom-up in game coordinates converted to Pygame top-down)
        x_origin = self.x + 325
        y_origin = self.y + 40
        h_painting = self.height - 80

        scale_h_in = h_painting / max(1, num_inputs - 1)
        scale_h_hid = h_painting / max(1, num_hidden - 1)
        scale_w = (self.width - 475) / 2.0
        neuron_radius = 8 if num_hidden > 8 else 10

        font = assets.font_small or pygame.font.SysFont("Arial", 13)

        # 1. Draw input labels and values
        labels = [
            f"[Obstáculo] Distancia: {sensor_values[0]:.1f}",
            f"[Obstáculo] Largura: {sensor_values[1]:.1f}",
            f"[Obstáculo] Altura: {sensor_values[2]:.1f}",
            f"[Obstáculo] Comprimento: {sensor_values[3]:.1f}",
            f"[Cenário] Velocidade: {sensor_values[4]:.2f}",
            f"[Dinossauro] Altura: {sensor_values[5]:.1f}",
        ]

        in_coords: List[Tuple[float, float]] = []
        for i, text in enumerate(labels):
            pos_y = y_origin + i * scale_h_in
            pos_x = x_origin
            in_coords.append((pos_x, pos_y))

            txt_surf = font.render(text, True, text_color)
            surface.blit(txt_surf, (self.x + 15, pos_y - 8))

        # 2. Draw output labels
        out_labels = ["Pular", "Abaixar", "Avião"]
        out_coords: List[Tuple[float, float]] = []
        y_mid_in = y_origin + h_painting / 2.0
        scale_h_out = scale_h_in * 0.8
        y_out_start = y_mid_in - (scale_h_out * (num_outputs - 1)) / 2.0

        for i, text in enumerate(out_labels):
            pos_y = y_out_start + i * scale_h_out
            pos_x = x_origin + 2 * scale_w
            out_coords.append((pos_x, pos_y))

            is_active = (brain.last_outputs[i] > 0.15) if i < len(brain.last_outputs) else False
            color = (50, 220, 50) if is_active else inactive_out_color
            txt_surf = font.render(text, True, color)
            surface.blit(txt_surf, (self.x + self.width - 90, pos_y - 8))

        # Hidden neuron coordinates
        hid_coords: List[Tuple[float, float]] = []
        for j in range(num_hidden):
            pos_y = y_origin + j * scale_h_hid
            pos_x = x_origin + scale_w
            hid_coords.append((pos_x, pos_y))

        # 3. Draw synaptic connection lines
        # Inputs -> Hidden
        for j, (hx, hy) in enumerate(hid_coords):
            for i, (ix, iy) in enumerate(in_coords):
                weight = brain.weights_hidden[j, i]
                self._draw_synapse(surface, (ix, iy), (hx, hy), weight)

        # Hidden -> Outputs
        for o, (ox, oy) in enumerate(out_coords):
            for j, (hx, hy) in enumerate(hid_coords):
                weight = brain.weights_output[o, j]
                self._draw_synapse(surface, (hx, hy), (ox, oy), weight)

        # 4. Draw Input Neurons
        for i, (ix, iy) in enumerate(in_coords):
            act = brain.last_inputs[i] if i < len(brain.last_inputs) else 0.0
            self._draw_neuron_node(surface, int(ix), int(iy), neuron_radius, act > 0, (200, 200, 255))
            # Arrow
            if assets.arrow_sprite:
                surface.blit(assets.arrow_sprite, (int(ix) - 40, int(iy) - 6))

        # 5. Draw Hidden Neurons
        for j, (hx, hy) in enumerate(hid_coords):
            act = brain.last_hidden[j] if j < len(brain.last_hidden) else 0.0
            self._draw_neuron_node(surface, int(hx), int(hy), neuron_radius, act > 0.05, (255, 230, 150))

        # 6. Draw Output Neurons
        for o, (ox, oy) in enumerate(out_coords):
            act = brain.last_outputs[o] if o < len(brain.last_outputs) else 0.0
            self._draw_neuron_node(surface, int(ox), int(oy), neuron_radius, act > 0.15, (150, 255, 150))

    def _draw_synapse(
        self,
        surface: pygame.Surface,
        start: Tuple[float, float],
        end: Tuple[float, float],
        weight: float,
    ) -> None:
        """Renders connection with color based on sign and thickness on magnitude."""
        if abs(weight) < 1e-4:
            return

        mag = min(3.5, max(1.0, abs(weight) / 250.0))
        if weight > 0:
            color = (60, 120, 220, 130)  # Blue/cyan for positive excitatory
        else:
            color = (220, 60, 60, 130)   # Red for negative inhibitory

        pygame.draw.line(surface, color, start, end, int(mag))

    def _draw_neuron_node(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        radius: int,
        is_active: bool,
        active_color: Tuple[int, int, int],
    ) -> None:
        """Renders a single neuron circle with glow effect if active."""
        if is_active:
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 255, 200, 60), (radius * 2, radius * 2), radius * 2)
            surface.blit(glow_surf, (x - radius * 2, y - radius * 2))

            pygame.draw.circle(surface, active_color, (x, y), radius)
            pygame.draw.circle(surface, (255, 255, 255), (x, y), max(1, radius - 3))
        else:
            pygame.draw.circle(surface, (200, 200, 200), (x, y), radius)
            pygame.draw.circle(surface, (120, 120, 120), (x, y), radius, 2)
