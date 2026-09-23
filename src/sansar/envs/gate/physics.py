"""Gate-env physics.

One vectorised `step_batch` is the single source of truth; `GateEngine` is a
thin n=1 wrapper implementing the Engine contract. There is no second scalar
implementation to drift out of sync.

Base car dynamics (steering, self-centering, acceleration, road-edge clamp)
are identical to `envs/driving/physics.ClassicEngine` — `tests/test_gate.py`
asserts step-for-step equality with it when the gate mechanic is inactive, so
V0.5 results are not confounded by a changed vehicle model.
"""

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from sansar.envs.gate.world import (
    BIT,
    BIT_CLOSED,
    BIT_OPEN,
    COLLIDED,
    DISTANCE,
    HEADING,
    SPEED,
    STATE_DIM,
    X,
)

STEER = np.array([0.0, -1.0, 1.0])  # NONE, LEFT, RIGHT


@dataclass(frozen=True)
class CarParams:
    dt: float
    half_width: float
    half_length: float
    road_half_width: float
    initial_speed: float
    max_speed: float
    accel: float
    steer_rate: float
    max_heading: float
    self_center_rate: float
    collision_speed_factor: float
    collision_min_speed: float
    gate_half_length: float
    gate_slow_factor: float

    @classmethod
    def from_cfg(cls, env_cfg: DictConfig) -> "CarParams":
        car = env_cfg.car
        return cls(
            dt=1.0 / float(env_cfg.sim_hz),
            half_width=float(car.half_width),
            half_length=float(car.half_length),
            road_half_width=float(env_cfg.road.half_width),
            initial_speed=float(car.initial_speed),
            max_speed=float(car.max_speed),
            accel=float(car.accel),
            steer_rate=float(car.steer_rate),
            max_heading=float(car.max_heading),
            self_center_rate=float(car.self_center_rate),
            collision_speed_factor=float(car.collision_speed_factor),
            collision_min_speed=float(car.collision_min_speed),
            gate_half_length=float(env_cfg.gate.half_length),
            gate_slow_factor=float(env_cfg.gate.slow_factor),
        )


def step_batch(
    states: np.ndarray,      # (N, STATE_DIM)
    actions: np.ndarray,     # (N,) int
    p: CarParams,
    pad_distance: np.ndarray,   # (N,)
    gate_distance: np.ndarray,  # (N,)
    heading_noise: np.ndarray | None = None,  # (N,) rad added to heading
) -> np.ndarray:
    """Advance N independent gate-world episodes by one fixed timestep.

    `heading_noise` makes the dynamics STOCHASTIC, which is a correctness
    requirement for this benchmark rather than a realism flourish. In a
    deterministic simulator the state->state map is injective, so the exact
    float value of `x` is a hash of the entire trajectory and a sufficiently
    flexible one-step model can read the hidden bit straight out of its low
    order bits. Measured: with deterministic dynamics, `frac_open` swung
    between 0.04 and 0.97 across bins of `x` 0.7 MM apart, and a memoryless
    MLP scored 0.73 on a task whose information-theoretic ceiling is 0.50.
    Process noise makes the map many-to-one, so the information is destroyed
    at the source instead of merely being decorrelated.

    Noise is supplied by the caller rather than drawn here, so `step_batch`
    stays a pure function of its inputs and episodes remain exactly
    replayable from (states, actions, noise).
    """
    x, heading, speed, distance = (
        states[:, X].copy(),
        states[:, HEADING].copy(),
        states[:, SPEED].copy(),
        states[:, DISTANCE].copy(),
    )
    bit = states[:, BIT].copy()

    steer = STEER[actions]
    turning = steer != 0.0
    decay = p.self_center_rate * p.dt
    heading = np.where(
        turning,
        heading + steer * p.steer_rate * p.dt,
        heading - np.copysign(np.minimum(np.abs(heading), decay), heading),
    )
    if heading_noise is not None:
        heading = heading + heading_noise
    np.clip(heading, -p.max_heading, p.max_heading, out=heading)

    speed = np.minimum(speed + p.accel * p.dt, p.max_speed)

    prev_distance = distance
    x = x + speed * np.sin(heading) * p.dt
    distance = distance + speed * np.cos(heading) * p.dt

    # road is straight in the gate env: centerline is 0 everywhere
    limit = p.road_half_width - p.half_width
    collided = np.abs(x) > limit
    x = np.where(collided, np.copysign(limit, x), x)
    speed = np.where(
        collided, np.maximum(speed * p.collision_speed_factor, p.collision_min_speed), speed
    )

    # pad line: crossing it sets the hidden bit from the side we crossed on.
    # LEFT (x < 0) -> OPEN, RIGHT (x >= 0) -> CLOSED.
    crossed_pad = (prev_distance < pad_distance) & (distance >= pad_distance)
    bit = np.where(crossed_pad, np.where(x < 0.0, BIT_OPEN, BIT_CLOSED), bit)

    # gate: spans the full road. Blocks only when the bit says CLOSED.
    at_gate = np.abs(distance - gate_distance) < (p.half_length + p.gate_half_length)
    blocked = at_gate & (bit == BIT_CLOSED)
    speed = np.where(
        blocked, np.maximum(speed * p.gate_slow_factor, p.collision_min_speed), speed
    )
    collided = collided | blocked

    out = np.empty_like(states)
    out[:, X] = x
    out[:, HEADING] = heading
    out[:, SPEED] = speed
    out[:, DISTANCE] = distance
    out[:, COLLIDED] = collided.astype(states.dtype)
    out[:, BIT] = bit
    return out


class GateEngine:
    """n=1 wrapper implementing the Engine contract on raw state arrays."""

    def __init__(self, env_cfg: DictConfig, pad_distance: float, gate_distance: float):
        self.p = CarParams.from_cfg(env_cfg)
        self.pad_distance = np.array([pad_distance])
        self.gate_distance = np.array([gate_distance])

    def reset(self) -> np.ndarray:
        s = np.zeros(STATE_DIM)
        s[SPEED] = self.p.initial_speed
        s[BIT] = BIT_CLOSED
        return s

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        return step_batch(
            state[None, :].astype(np.float64),
            np.array([int(action)]),
            self.p,
            self.pad_distance,
            self.gate_distance,
        )[0]
