"""Blockers on the road.

Layout is a pure deterministic function of (seed, index): like road geometry,
obstacles are static world structure rather than dynamic state, so the
neural engine's prediction target stays just the car. (Mutable obstacles —
ones that disappear when hit — come later as the persistent-state
experiments.)
"""

import math
from dataclasses import dataclass

from omegaconf import DictConfig

from sansar.envs.driving.road import Road


def _rand01(seed: int, index: int, salt: float) -> float:
    """Deterministic pseudo-random in [0, 1) per (seed, index, salt)."""
    v = math.sin(index * 12.9898 + seed * 78.233 + salt * 37.719) * 43758.5453
    return v - math.floor(v)


@dataclass(frozen=True)
class Obstacle:
    distance: float     # forward position of center (m)
    x: float            # lateral position of center (m, world frame)
    half_width: float
    half_length: float


class Obstacles:
    def __init__(self, cfg: DictConfig, road: Road):
        self.enabled = bool(cfg.enabled)
        self.seed = int(cfg.seed)
        self.first_at = float(cfg.first_at)
        self.spacing = float(cfg.spacing)
        self.jitter = float(cfg.jitter)
        self.half_width = float(cfg.half_width)
        self.half_length = float(cfg.half_length)
        self.edge_margin = float(cfg.edge_margin)
        self.road = road

    def _at_index(self, i: int) -> Obstacle:
        d = self.first_at + i * self.spacing + (_rand01(self.seed, i, 1.0) - 0.5) * self.jitter
        span = self.road.half_width - self.half_width - self.edge_margin
        offset = (_rand01(self.seed, i, 2.0) - 0.5) * 2.0 * max(span, 0.0)
        return Obstacle(
            distance=d,
            x=self.road.center_at(d) + offset,
            half_width=self.half_width,
            half_length=self.half_length,
        )

    def in_range(self, d0: float, d1: float) -> list[Obstacle]:
        if not self.enabled:
            return []
        lo = max(0, math.floor((d0 - self.first_at) / self.spacing) - 1)
        hi = math.ceil((d1 - self.first_at) / self.spacing) + 1
        return [
            ob
            for ob in (self._at_index(i) for i in range(lo, hi + 1))
            if d0 <= ob.distance <= d1
        ]
