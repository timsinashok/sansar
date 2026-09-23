"""Gate-world geometry and state.

The gate env adds one mechanic to V0 driving: a *pad line* whose crossing
side sets a hidden bit, and a *gate* G steps later whose behaviour depends on
that bit. Nothing else changes.

State is (x, heading, speed, distance, collided, gate_bit) — six numbers.
`gate_bit` is part of the ground-truth world state, which keeps the Engine
contract pure, but it is deliberately EXCLUDED from the observation the model
sees (`sansar.models.gate_features`). That gap is the experiment: the world
is fully determined, the observation is not.

Layout of one trial:

    d=0            pad_distance                 gate_distance
     |                  |                             |
     +------------------+------ G steps of driving ---+
       approach          ^                      ^
                    x<0 -> OPEN            funnel: policy centres
                    x>=0 -> CLOSED         the car for funnel_m before
                                           the gate, so gate-time x
                                           carries no trace of the pad
"""

from dataclasses import dataclass

import numpy as np

# state layout
X, HEADING, SPEED, DISTANCE, COLLIDED, BIT = range(6)
STATE_DIM = 6

BIT_CLOSED = 0.0
BIT_OPEN = 1.0


@dataclass(frozen=True)
class GateLayout:
    """Placement of one trial's pad line and gate."""

    pad_distance: float
    gate_distance: float

    @property
    def gap_m(self) -> float:
        return self.gate_distance - self.pad_distance


def layout_for_gap(
    target_gap_steps: int,
    cruise_speed: float,
    dt: float,
    pad_distance: float,
) -> GateLayout:
    """Place the gate so the pad->gate gap is ~target_gap_steps at cruise.

    The realised gap in steps is measured per episode at collection time
    (speed is not exactly cruise), and results are reported against the
    realised median.
    """
    return GateLayout(
        pad_distance=float(pad_distance),
        gate_distance=float(pad_distance + target_gap_steps * cruise_speed * dt),
    )


def initial_state(n: int) -> np.ndarray:
    """(n, STATE_DIM) fresh episode states."""
    s = np.zeros((n, STATE_DIM), dtype=np.float64)
    s[:, BIT] = BIT_CLOSED
    return s
