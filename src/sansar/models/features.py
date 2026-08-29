"""Model input features.

The raw car state contains absolute `distance`, which is unbounded and
useless as a network input. Instead each timestep is encoded as LOCAL
features computed from the state plus the static world geometry (road
centerline, blocker layout). Geometry is part of the world definition — like
the renderer — so conditioning on it is fair; the dynamics are what's
learned.

Per-step features (FEATURE_DIM = 10), before normalization:
    x_rel     lateral offset from the road centerline
    heading, speed, collided
    dd1, dx1  next blocker: forward gap (capped) and lateral offset from car
    dd2, dx2  the blocker after that
    curve1/2  centerline shift 5 m / 15 m ahead (zero on straight roads)

Action is appended as a one-hot (3) after normalization → INPUT_DIM = 13.
"""

import math

import torch
from omegaconf import DictConfig

from sansar.envs.driving.obstacles import Obstacles
from sansar.envs.driving.road import Road

FEATURE_DIM = 10
N_ACTIONS = 3
INPUT_DIM = FEATURE_DIM + N_ACTIONS

OBSTACLE_CAP = 30.0   # meters: blockers further than this read as "none"
LOOKAHEADS = (5.0, 15.0)


class Geometry:
    """Torch-side static world geometry: road centerline + blocker layout."""

    def __init__(self, env_cfg: DictConfig, device: str, max_distance: float = 20000.0):
        road = Road(env_cfg.road)
        self.kind = road.kind
        self.amplitude = road._amplitude
        self.wavelength = road._wavelength
        self.device = device

        obs = Obstacles(env_cfg.obstacles, road)
        obs_list = obs.in_range(0.0, max_distance)
        self.n_obs = len(obs_list)
        if self.n_obs:
            self.ob_d = torch.tensor([o.distance for o in obs_list], device=device)
            self.ob_x = torch.tensor([o.x for o in obs_list], device=device)
        else:
            self.ob_d = torch.zeros(0, device=device)
            self.ob_x = torch.zeros(0, device=device)

    def center(self, d: torch.Tensor) -> torch.Tensor:
        if self.kind == "straight":
            return torch.zeros_like(d)
        return self.amplitude * torch.sin(2 * math.pi * d / self.wavelength)

    def _blocker(self, idx: torch.Tensor, d: torch.Tensor, x: torch.Tensor):
        valid = idx < self.n_obs
        idx_c = idx.clamp(max=max(self.n_obs - 1, 0))
        dd = torch.where(valid, self.ob_d[idx_c] - d, torch.full_like(d, OBSTACLE_CAP))
        dx = torch.where(valid, self.ob_x[idx_c] - x, torch.zeros_like(x))
        return dd.clamp(-2.0, OBSTACLE_CAP), dx

    def features(self, states: torch.Tensor) -> torch.Tensor:
        """states (..., 5) -> raw features (..., FEATURE_DIM)."""
        x, heading, speed, d, collided = states.unbind(-1)
        ctr = self.center(d)
        x_rel = x - ctr
        curves = [self.center(d + la) - ctr for la in LOOKAHEADS]

        if self.n_obs:
            idx = torch.searchsorted(self.ob_d, (d - 2.0).contiguous())
            dd1, dx1 = self._blocker(idx, d, x)
            dd2, dx2 = self._blocker(idx + 1, d, x)
        else:
            dd1 = dd2 = torch.full_like(d, OBSTACLE_CAP)
            dx1 = dx2 = torch.zeros_like(d)

        return torch.stack(
            [x_rel, heading, speed, collided, dd1, dx1, dd2, dx2, *curves], dim=-1
        )


def compute_stats(geom: Geometry, states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Normalization stats from training states (N, 5): feature mean/std and
    per-dim stats of the one-step deltas of (x, heading, speed, distance)."""
    feats = geom.features(states)
    deltas = states[1:, :4] - states[:-1, :4]
    return {
        "feat_mean": feats.mean(0),
        "feat_std": feats.std(0).clamp(min=1e-4),
        "delta_mean": deltas.mean(0),
        "delta_std": deltas.std(0).clamp(min=1e-4),
    }


def tokens(
    geom: Geometry,
    stats: dict[str, torch.Tensor],
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """(B, K, 5) states + (B, K) int actions -> (B, K, INPUT_DIM) inputs."""
    feats = (geom.features(states) - stats["feat_mean"]) / stats["feat_std"]
    onehot = torch.nn.functional.one_hot(actions.long(), N_ACTIONS).to(feats.dtype)
    return torch.cat([feats, onehot], dim=-1)


def integrate(states_last: torch.Tensor, pred: torch.Tensor, stats) -> torch.Tensor:
    """Apply a model prediction (4 normalized deltas + collision logit) to the
    last absolute state (B, 5) -> next absolute state (B, 5)."""
    delta = pred[:, :4] * stats["delta_std"] + stats["delta_mean"]
    nxt4 = states_last[:, :4] + delta
    collided = (pred[:, 4] > 0).to(states_last.dtype)
    return torch.cat([nxt4, collided.unsqueeze(-1)], dim=-1)
