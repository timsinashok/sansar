"""Trajectory collection: roll scripted policies through the classic engine
and log episodes to disk.

Each episode file ep_XXXXX.npz holds:
    states:  (steps+1, CarState.DIM) float32 — states[t] is the state BEFORE
             actions[t]; states[t+1] the result
    actions: (steps,) int8
    policy:  the driver personality that produced it

The run directory also gets a snapshot of the fully-resolved config, so a
dataset is always reproducible from (config, data.seed) alone.
"""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from sansar.core.types import Action, CarState
from sansar.data.policies import make_policy
from sansar.envs.driving.physics import ClassicEngine
from sansar.utils.config import save_config


def collect(cfg: DictConfig) -> Path:
    out = Path(cfg.data.out_dir) / cfg.data.run_name
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    engine = ClassicEngine(cfg.env)
    rng = np.random.default_rng(int(cfg.data.seed))
    names = list(cfg.data.policy_mix.keys())
    weights = np.array([float(cfg.data.policy_mix[n]) for n in names])
    weights = weights / weights.sum()

    episodes = int(cfg.data.episodes)
    steps = int(cfg.data.steps)
    collision_steps = 0

    for ep in range(episodes):
        name = str(rng.choice(names, p=weights))
        policy = make_policy(name, engine, np.random.default_rng(rng.integers(2**31)))

        state = engine.reset()
        states = np.empty((steps + 1, CarState.DIM), dtype=np.float32)
        actions = np.empty(steps, dtype=np.int8)
        states[0] = state.to_array()
        for t in range(steps):
            a = policy.act(state, t)
            actions[t] = int(a)
            state = engine.step(state, a)
            states[t + 1] = state.to_array()

        collision_steps += int(states[:, 4].sum())
        np.savez_compressed(out / f"ep_{ep:05d}.npz", states=states, actions=actions, policy=name)
        if (ep + 1) % 25 == 0 or ep + 1 == episodes:
            print(f"[collect] {ep + 1}/{episodes} episodes")

    total = episodes * steps
    print(
        f"[collect] done: {total} transitions in {out}/ "
        f"({100 * collision_steps / total:.1f}% collision steps)"
    )
    return out


def replay(states: np.ndarray, actions: np.ndarray, engine: ClassicEngine) -> np.ndarray:
    """Re-simulate an episode from its initial state and recorded actions.
    Must reproduce `states` exactly — the determinism check behind every
    dataset."""
    out = np.empty_like(states)
    s = CarState.from_array(states[0])
    out[0] = s.to_array()
    for t, a in enumerate(actions):
        s = engine.step(s, Action(int(a)))
        out[t + 1] = s.to_array()
    return out
