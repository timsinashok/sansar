"""The interactive game loop.

Runs the simulation at env.sim_hz with a fixed timestep and renders every
(sim_hz // render_hz)-th state — the sim/visualization frequency split from
the core idea. Takes any Engine, so the neural engine drops in unchanged.
"""

import pygame
from omegaconf import DictConfig

from sansar.core.engine import Engine
from sansar.core.types import Action
from sansar.envs.driving.obstacles import Obstacles
from sansar.envs.driving.road import Road
from sansar.render.pygame_renderer import PygameRenderer


def _read_action() -> Action:
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    if left and not right:
        return Action.LEFT
    if right and not left:
        return Action.RIGHT
    return Action.NONE


def run(cfg: DictConfig, engine: Engine, title: str = "sansar — classic engine") -> None:
    sim_hz = int(cfg.env.sim_hz)
    steps_per_frame = max(1, sim_hz // int(cfg.env.render_hz))

    road = Road(cfg.env.road)
    obstacles = Obstacles(cfg.env.obstacles, road)
    renderer = PygameRenderer(cfg.render, road, obstacles, float(cfg.env.car.half_width), title)
    clock = pygame.time.Clock()

    state = engine.reset()
    step = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = engine.reset()
                    step = 0

        state = engine.step(state, _read_action())
        step += 1
        if step % steps_per_frame == 0:
            renderer.draw(state)
        clock.tick(sim_hz)

    renderer.close()


def run_duel(cfg: DictConfig, neural: Engine, truth: Engine) -> None:
    """Side-by-side mode: the neural engine is the game (solid car, camera),
    the classic engine runs the same inputs as ground truth (outline ghost).
    The gap between them IS the divergence, live on screen."""
    sim_hz = int(cfg.env.sim_hz)
    steps_per_frame = max(1, sim_hz // int(cfg.env.render_hz))

    road = Road(cfg.env.road)
    obstacles = Obstacles(cfg.env.obstacles, road)
    renderer = PygameRenderer(
        cfg.render, road, obstacles, float(cfg.env.car.half_width),
        "sansar — duel: neural (solid) vs classic ghost",
    )
    clock = pygame.time.Clock()

    ns, ts = neural.reset(), truth.reset()
    step = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    ns, ts = neural.reset(), truth.reset()
                    step = 0

        action = _read_action()
        ns = neural.step(ns, action)
        ts = truth.step(ts, action)
        step += 1
        if step % steps_per_frame == 0:
            gap = f"drift  x {abs(ns.x - ts.x):5.2f} m   dist {abs(ns.distance - ts.distance):5.2f} m"
            renderer.draw(ns, ghost=ts, hud_extra=gap)
        clock.tick(sim_hz)

    renderer.close()
