"""Procedural 8-bit Retro Sound Synthesizer using NumPy and Pygame Mixer."""

import numpy as np
import pygame
from typing import Optional


class SoundSynthesizer:
    """
    Generates procedural chiptune sound effects in real time without external audio files.
    Produces 16-bit stereo buffers compatible with pygame.mixer.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.muted = False
        self.initialized = False

        self.snd_jump: Optional[pygame.mixer.Sound] = None
        self.snd_airplane: Optional[pygame.mixer.Sound] = None
        self.snd_death: Optional[pygame.mixer.Sound] = None
        self.snd_score: Optional[pygame.mixer.Sound] = None

    def initialize(self) -> None:
        """Initializes mixer and synthesizes retro sound effects."""
        if self.initialized:
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=512)

            self.snd_jump = self._synth_jump()
            self.snd_airplane = self._synth_airplane()
            self.snd_death = self._synth_death()
            self.snd_score = self._synth_score()

            self.initialized = True
        except Exception as e:
            print(f"[Aviso] Áudio desativado ou indisponível: {e}")
            self.muted = True

    def toggle_mute(self) -> bool:
        """Toggles audio mute on/off. Returns current mute state."""
        self.muted = not self.muted
        return self.muted

    def _to_stereo_sound(self, mono_samples: np.ndarray) -> pygame.mixer.Sound:
        """Converts 1D float array [-1.0, 1.0] to 2D int16 stereo sound buffer."""
        int16_samples = (np.clip(mono_samples, -1.0, 1.0) * 16000).astype(np.int16)
        stereo_samples = np.column_stack([int16_samples, int16_samples])
        return pygame.sndarray.make_sound(stereo_samples)

    def _synth_jump(self) -> pygame.mixer.Sound:
        """Ascending frequency chirp for jump."""
        duration = 0.12
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)

        # Frequency sweeps from 250Hz to 650Hz
        freq = np.linspace(250, 650, n_samples)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        # Square wave with envelope
        wave = np.sign(np.sin(phase)) * np.linspace(1.0, 0.2, n_samples)
        return self._to_stereo_sound(wave)

    def _synth_airplane(self) -> pygame.mixer.Sound:
        """Aerodynamic wind swoosh for airplane."""
        duration = 0.2
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)

        # Modulated noise
        noise = np.random.uniform(-0.5, 0.5, n_samples)
        sine = np.sin(2 * np.pi * 120 * t)
        envelope = np.sin(np.pi * t / duration)

        wave = (noise * 0.7 + sine * 0.3) * envelope
        return self._to_stereo_sound(wave)

    def _synth_death(self) -> pygame.mixer.Sound:
        """Crunchy descending buzz for death collision."""
        duration = 0.25
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)

        freq = np.linspace(180, 50, n_samples)
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate

        noise = np.random.uniform(-0.4, 0.4, n_samples)
        square = np.sign(np.sin(phase))
        envelope = np.linspace(1.0, 0.0, n_samples) ** 1.5

        wave = (square * 0.6 + noise * 0.4) * envelope
        return self._to_stereo_sound(wave)

    def _synth_score(self) -> pygame.mixer.Sound:
        """Two-tone bell chime for 1000-distance milestone."""
        duration = 0.18
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)

        # Tone 1 (880Hz), then Tone 2 (1174Hz)
        half = n_samples // 2
        freqs = np.zeros(n_samples)
        freqs[:half] = 880.0
        freqs[half:] = 1174.66

        phase = 2 * np.pi * np.cumsum(freqs) / self.sample_rate
        env = np.ones(n_samples)
        env[:half] = np.linspace(1.0, 0.5, half)
        env[half:] = np.linspace(1.0, 0.0, n_samples - half)

        wave = np.sin(phase) * env
        return self._to_stereo_sound(wave)

    def play_jump(self) -> None:
        if self.initialized and not self.muted and self.snd_jump:
            self.snd_jump.play()

    def play_airplane(self) -> None:
        if self.initialized and not self.muted and self.snd_airplane:
            self.snd_airplane.play()

    def play_death(self) -> None:
        if self.initialized and not self.muted and self.snd_death:
            self.snd_death.play()

    def play_score(self) -> None:
        if self.initialized and not self.muted and self.snd_score:
            self.snd_score.play()
