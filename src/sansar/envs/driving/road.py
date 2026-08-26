"""Road geometry. The centerline is a function of forward distance, so
straight vs. curved roads (V0 vs. V0.1) are a config switch, not new code."""

import math

from omegaconf import DictConfig


class Road:
    def __init__(self, cfg: DictConfig):
        self.kind = cfg.kind
        self.half_width = float(cfg.half_width)
        self._amplitude = float(cfg.curve_amplitude)
        self._wavelength = float(cfg.curve_wavelength)

    def center_at(self, distance: float) -> float:
        if self.kind == "straight":
            return 0.0
        return self._amplitude * math.sin(2 * math.pi * distance / self._wavelength)
