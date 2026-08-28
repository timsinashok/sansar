"""Scripted driver policies for data collection.

Each policy is a different driving personality; together they cover the state
space the neural engine must learn — including boundaries, blocker
collisions, and erratic inputs. All randomness comes from the injected numpy
Generator, so a (config, seed) pair reproduces the exact dataset.
"""

import math

import numpy as np

from sansar.core.types import Action, CarState
from sansar.envs.driving.physics import ClassicEngine


class _Base:
    def __init__(self, engine: ClassicEngine, rng: np.random.Generator):
        self.engine = engine
        self.rng = rng

    def act(self, state: CarState, t: int) -> Action:
        raise NotImplementedError

    def _steer_toward(self, state: CarState, target_x: float) -> Action:
        """Bang-bang controller: pick the action whose heading change moves
        the car toward target_x, with a deadband to avoid chatter."""
        error = target_x - state.x
        desired_heading = max(-0.4, min(0.4, 0.5 * error))
        if desired_heading > state.heading + 0.02:
            return Action.RIGHT
        if desired_heading < state.heading - 0.02:
            return Action.LEFT
        return Action.NONE


class LanePolicy(_Base):
    """Follows the road center with noise; steers around blockers ahead.
    The closest thing to typical human play."""

    def __init__(self, engine, rng):
        super().__init__(engine, rng)
        self.noise_p = float(rng.uniform(0.0, 0.08))

    def act(self, state: CarState, t: int) -> Action:
        if self.rng.random() < self.noise_p:
            return Action(self.rng.integers(0, 3))

        lookahead = max(4.0, state.speed * 1.0)
        d_ahead = state.distance + lookahead
        target = self.engine.road.center_at(d_ahead)

        clearance = self.engine.car_half_width + 0.5
        for ob in self.engine.obstacles.in_range(state.distance, d_ahead + 5.0):
            if abs(target - ob.x) < ob.half_width + clearance:
                center = self.engine.road.center_at(ob.distance)
                room_left = (ob.x - ob.half_width) - (center - self.engine.road.half_width)
                room_right = (center + self.engine.road.half_width) - (ob.x + ob.half_width)
                side = -1.0 if room_left > room_right else 1.0
                target = ob.x + side * (ob.half_width + clearance)
                break
        return self._steer_toward(state, target)


class RandomPolicy(_Base):
    """Sticky random actions: hold each sampled action for a while, since
    50 Hz coin flips average out to driving straight."""

    def __init__(self, engine, rng):
        super().__init__(engine, rng)
        self.action = Action.NONE
        self.until = 0

    def act(self, state: CarState, t: int) -> Action:
        if t >= self.until:
            self.action = Action(self.rng.choice([0, 1, 2], p=[0.4, 0.3, 0.3]))
            self.until = t + int(self.rng.integers(5, 30))
        return self.action


class SwervePolicy(_Base):
    """Sinusoidal steering — sweeps through the car's full dynamic range."""

    def __init__(self, engine, rng):
        super().__init__(engine, rng)
        self.period = float(rng.uniform(40, 120))  # steps per full swerve
        self.phase = float(rng.uniform(0, 2 * math.pi))

    def act(self, state: CarState, t: int) -> Action:
        wave = math.sin(2 * math.pi * t / self.period + self.phase)
        if wave > 0.2:
            return Action.RIGHT
        if wave < -0.2:
            return Action.LEFT
        return Action.NONE


class WallPolicy(_Base):
    """Alternates between aiming at a road edge and recovering to center —
    oversamples the rare collision dynamics."""

    def __init__(self, engine, rng):
        super().__init__(engine, rng)
        self.cycle = int(rng.integers(100, 250))
        self.attack_frac = float(rng.uniform(0.3, 0.5))
        self.side = 1.0 if rng.random() < 0.5 else -1.0

    def act(self, state: CarState, t: int) -> Action:
        phase = (t % self.cycle) / self.cycle
        if phase < self.attack_frac:
            target = self.engine.road.center_at(state.distance) + self.side * (
                self.engine.road.half_width + 1.0
            )
        else:
            target = self.engine.road.center_at(state.distance + state.speed)
        if phase < 0.01:
            self.side = 1.0 if self.rng.random() < 0.5 else -1.0
        return self._steer_toward(state, target)


POLICIES = {
    "lane": LanePolicy,
    "random": RandomPolicy,
    "swerve": SwervePolicy,
    "wall": WallPolicy,
}


def make_policy(name: str, engine: ClassicEngine, rng: np.random.Generator) -> _Base:
    return POLICIES[name](engine, rng)
