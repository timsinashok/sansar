"""State and action definitions shared by every engine (classic and neural)."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


@dataclass(frozen=True)
class CarState:
    x: float          # lateral position (m), 0 = world centerline
    heading: float    # rad, 0 = straight ahead, positive = rightward
    speed: float      # m/s along heading
    distance: float   # total forward progress (m)
    collided: bool    # scraping the road edge this step

    DIM = 5

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.heading, self.speed, self.distance, float(self.collided)],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, a: np.ndarray) -> "CarState":
        return cls(
            x=float(a[0]),
            heading=float(a[1]),
            speed=float(a[2]),
            distance=float(a[3]),
            collided=bool(a[4] > 0.5),
        )
