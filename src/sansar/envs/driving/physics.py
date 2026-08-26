"""The classic (hand-written) engine for the V0 driving env.

This is the ground-truth simulator: it drives the playable game, generates
training trajectories, and is the reference the neural engine is measured
against.
"""

import math

from omegaconf import DictConfig

from sansar.core.types import Action, CarState
from sansar.envs.driving.obstacles import Obstacles
from sansar.envs.driving.road import Road

_STEER = {Action.NONE: 0.0, Action.LEFT: -1.0, Action.RIGHT: 1.0}


class ClassicEngine:
    def __init__(self, env_cfg: DictConfig):
        self.dt = 1.0 / float(env_cfg.sim_hz)
        self.road = Road(env_cfg.road)
        self.obstacles = Obstacles(env_cfg.obstacles, self.road)
        car = env_cfg.car
        self.car_half_width = float(car.half_width)
        self.car_half_length = float(car.half_length)
        self.initial_speed = float(car.initial_speed)
        self.max_speed = float(car.max_speed)
        self.accel = float(car.accel)
        self.steer_rate = float(car.steer_rate)
        self.max_heading = float(car.max_heading)
        self.self_center_rate = float(car.self_center_rate)
        self.collision_speed_factor = float(car.collision_speed_factor)
        self.collision_min_speed = float(car.collision_min_speed)
        self.obstacle_slow_factor = float(env_cfg.obstacles.slow_factor)

    def reset(self) -> CarState:
        return CarState(
            x=self.road.center_at(0.0),
            heading=0.0,
            speed=self.initial_speed,
            distance=0.0,
            collided=False,
        )

    def step(self, state: CarState, action: Action) -> CarState:
        steer = _STEER[action]
        if steer != 0.0:
            heading = state.heading + steer * self.steer_rate * self.dt
        else:
            # self-center: heading decays toward straight when not steering
            decay = self.self_center_rate * self.dt
            heading = state.heading - math.copysign(min(abs(state.heading), decay), state.heading)
        heading = max(-self.max_heading, min(self.max_heading, heading))

        speed = min(state.speed + self.accel * self.dt, self.max_speed)

        x = state.x + speed * math.sin(heading) * self.dt
        distance = state.distance + speed * math.cos(heading) * self.dt

        # road-edge collision: clamp to the edge and slow down
        center = self.road.center_at(distance)
        limit = self.road.half_width - self.car_half_width
        collided = abs(x - center) > limit
        if collided:
            x = center + math.copysign(limit, x - center)
            speed = max(speed * self.collision_speed_factor, self.collision_min_speed)

        # obstacle collision: heavy slowdown while overlapping a blocker
        reach = self.car_half_length + self.obstacles.half_length + 1.0
        for ob in self.obstacles.in_range(distance - reach, distance + reach):
            if (
                abs(distance - ob.distance) < self.car_half_length + ob.half_length
                and abs(x - ob.x) < self.car_half_width + ob.half_width
            ):
                speed = max(speed * self.obstacle_slow_factor, self.collision_min_speed)
                collided = True
                break

        return CarState(x=x, heading=heading, speed=speed, distance=distance, collided=collided)
