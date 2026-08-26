"""Pygame renderer for the driving env.

The renderer only reads CarState + Road — it never touches the engine, so it
draws the classic and neural simulations identically.
"""

import pygame
from omegaconf import DictConfig

from sansar.core.types import CarState
from sansar.envs.driving.obstacles import Obstacles
from sansar.envs.driving.road import Road

BG = (24, 26, 30)
ROAD = (52, 56, 62)
EDGE = (210, 210, 215)
POST = (250, 160, 60)
DASH = (120, 124, 130)
BLOCK = (90, 140, 235)
CAR = (235, 90, 70)
CAR_HIT = (255, 200, 60)
HUD = (200, 203, 208)


class PygameRenderer:
    def __init__(
        self,
        cfg: DictConfig,
        road: Road,
        obstacles: Obstacles,
        car_half_width: float,
        title: str = "sansar",
    ):
        pygame.init()
        self.w = int(cfg.window_width)
        self.h = int(cfg.window_height)
        self.ppm = float(cfg.pixels_per_meter)
        self.road = road
        self.obstacles = obstacles
        self.car_half_width = car_half_width
        self.car_screen_y = int(self.h * 0.78)
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("menlo", 16)

    def _to_screen_x(self, world_x: float) -> int:
        return int(self.w / 2 + world_x * self.ppm)

    def draw(self, state: CarState) -> None:
        self.screen.fill(BG)

        # road edges: sample world distance per screen row, car pinned at car_screen_y
        mpp = 1.0 / self.ppm
        left_pts, right_pts = [], []
        for y in range(0, self.h, 4):
            d = state.distance + (self.car_screen_y - y) * mpp
            c = self.road.center_at(d)
            left_pts.append((self._to_screen_x(c - self.road.half_width), y))
            right_pts.append((self._to_screen_x(c + self.road.half_width), y))
        pygame.draw.polygon(self.screen, ROAD, left_pts + right_pts[::-1])
        pygame.draw.lines(self.screen, EDGE, False, left_pts, 3)
        pygame.draw.lines(self.screen, EDGE, False, right_pts, 3)

        # dashed centerline, scrolling with distance
        dash_period_m = 2.0
        for y in range(0, self.h, 2):
            d = state.distance + (self.car_screen_y - y) * mpp
            if (d % dash_period_m) < dash_period_m / 2:
                cx = self._to_screen_x(self.road.center_at(d))
                self.screen.fill(DASH, (cx - 1, y, 3, 2))

        # roadside posts every 10 m — a strong speed cue that can't alias
        # away like the uniform dashes can
        post_period_m = 10.0
        d_top = state.distance + self.car_screen_y * mpp
        d_bottom = state.distance - (self.h - self.car_screen_y) * mpp
        d = (d_bottom // post_period_m) * post_period_m
        while d < d_top:
            y = int(self.car_screen_y - (d - state.distance) * self.ppm)
            if 0 <= y < self.h:
                c = self.road.center_at(d)
                lx = self._to_screen_x(c - self.road.half_width)
                rx = self._to_screen_x(c + self.road.half_width)
                self.screen.fill(POST, (lx - 8, y - 3, 6, 6))
                self.screen.fill(POST, (rx + 2, y - 3, 6, 6))
            d += post_period_m

        # blockers
        view_reach = self.h * mpp
        for ob in self.obstacles.in_range(state.distance - view_reach, state.distance + view_reach):
            ow = int(2 * ob.half_width * self.ppm)
            oh = int(2 * ob.half_length * self.ppm)
            rect = pygame.Rect(0, 0, ow, oh)
            rect.center = (
                self._to_screen_x(ob.x),
                int(self.car_screen_y - (ob.distance - state.distance) * self.ppm),
            )
            pygame.draw.rect(self.screen, BLOCK, rect, border_radius=3)

        # car
        car_w = int(2 * self.car_half_width * self.ppm)
        car_h = int(car_w * 1.8)
        rect = pygame.Rect(0, 0, car_w, car_h)
        rect.center = (self._to_screen_x(state.x), self.car_screen_y)
        pygame.draw.rect(self.screen, CAR_HIT if state.collided else CAR, rect, border_radius=4)

        hud = f"dist {state.distance:7.1f} m   speed {state.speed * 3.6:5.1f} km/h"
        self.screen.blit(self.font.render(hud, True, HUD), (12, 10))

        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
