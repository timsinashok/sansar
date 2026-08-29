"""Episode loading and window sampling for training.

The whole V0 dataset is tiny (600k x 5 floats ≈ 12 MB), so everything lives
in memory as two arrays. Sampling returns windows long enough for a K-step
context plus a k-step rollout target.
"""

from pathlib import Path

import numpy as np
import torch


class EpisodeData:
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = states     # (E, T+1, 5) float32
        self.actions = actions   # (E, T) int8

    @classmethod
    def load(cls, run_dir: str | Path) -> "EpisodeData":
        files = sorted(Path(run_dir).glob("ep_*.npz"))
        if not files:
            raise FileNotFoundError(f"no episodes in {run_dir} — run scripts/collect.py first")
        eps = [np.load(f) for f in files]
        return cls(
            np.stack([e["states"] for e in eps]),
            np.stack([e["actions"] for e in eps]),
        )

    def split(self, val_episodes: int) -> tuple["EpisodeData", "EpisodeData"]:
        v = max(1, int(val_episodes))
        return (
            EpisodeData(self.states[:-v], self.actions[:-v]),
            EpisodeData(self.states[-v:], self.actions[-v:]),
        )

    def flat_states(self) -> np.ndarray:
        return self.states.reshape(-1, self.states.shape[-1])

    def sample_windows(
        self, n: int, context: int, horizon: int, rng: np.random.Generator, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns states (n, context+horizon, 5) and actions
        (n, context+horizon-1): positions [0:context] are the model's window,
        positions [context:] are rollout targets."""
        length = context + horizon
        T = self.actions.shape[1]
        ep = rng.integers(0, self.states.shape[0], size=n)
        t0 = rng.integers(0, T - length + 1, size=n)
        idx = t0[:, None] + np.arange(length)[None, :]
        states = self.states[ep[:, None], idx]
        actions = self.actions[ep[:, None], idx[:, :-1]]
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(actions.astype(np.int64)).to(device),
        )
