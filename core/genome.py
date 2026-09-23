"""Genome module for DNA serialization, deserialization, and legacy format support."""

import struct
import json
from pathlib import Path
from typing import Union, List, Optional
import numpy as np


class Genome:
    """Represents a flat sequence of floating-point weights (DNA) for a Neural Network."""

    def __init__(self, weights: Union[np.ndarray, List[float]]):
        self.weights = np.array(weights, dtype=np.float64)

    @property
    def length(self) -> int:
        return len(self.weights)

    def copy(self) -> "Genome":
        return Genome(self.weights.copy())

    def to_legacy_binary(self, filepath: Union[str, Path]) -> None:
        """Saves weights matching original C format: int32 (dna_len) + N * double (float64)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(struct.pack("i", self.length))
            f.write(struct.pack(f"{self.length}d", *self.weights))

    @classmethod
    def from_legacy_binary(cls, filepath: Union[str, Path]) -> "Genome":
        """Loads weights from original C binary format."""
        path = Path(filepath)
        with open(path, "rb") as f:
            data = f.read(4)
            if len(data) < 4:
                raise ValueError(f"Corrupt or empty file at {filepath}")
            dna_len = struct.unpack("i", data)[0]
            expected_bytes = dna_len * 8
            raw_weights = f.read(expected_bytes)
            if len(raw_weights) < expected_bytes:
                raise ValueError(f"Incomplete weights in file {filepath}")
            weights = struct.unpack(f"{dna_len}d", raw_weights)
            return cls(weights)

    def to_json(self, filepath: Union[str, Path], metadata: Optional[dict] = None) -> None:
        """Saves weights and metadata to a readable JSON format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "length": self.length,
            "weights": self.weights.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "Genome":
        """Loads weights from JSON format."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return cls(data["weights"])

    @classmethod
    def random(cls, length: int, min_val: float = -1000.0, max_val: float = 1000.0) -> "Genome":
        """Generates random weights within specified range."""
        weights = np.random.uniform(min_val, max_val, size=length)
        return cls(weights)
