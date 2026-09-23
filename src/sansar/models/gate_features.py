"""Observation encoding for the gate env — the POMDP boundary.

Ground-truth state has six numbers; the observation has SIX FEATURES and
deliberately omits `gate_bit`. What the model can see:

    x, heading, speed, collided   ego state
    dpad                          signed gap to the pad line  (clipped +-30 m)
    dgate                         signed gap to the gate      (clipped +-30 m)

The hidden bit is computable from (dpad ~ 0, sign(x)) at a single instant and
is required at (dgate ~ 0), G steps later. Once the pad is more than 30 m
behind, `dpad` saturates and carries no information at all — so beyond
~30 m / (13.9 m/s * 0.02 s) ~= 108 steps the bit is recoverable ONLY from
whatever the model has retained internally.
"""

import torch

FEATURE_DIM = 6
N_ACTIONS = 3
INPUT_DIM = FEATURE_DIM + N_ACTIONS
OUTPUT_DIM = 5  # 4 normalised deltas (x, heading, speed, distance) + collision logit

GAP_CAP = 30.0

from sansar.envs.gate.world import COLLIDED, DISTANCE, HEADING, SPEED, X  # noqa: E402


def features(states: torch.Tensor, pad_d: torch.Tensor, gate_d: torch.Tensor) -> torch.Tensor:
    """states (B, T, 6), pad_d/gate_d (B,) -> raw features (B, T, FEATURE_DIM)."""
    x = states[..., X]
    heading = states[..., HEADING]
    speed = states[..., SPEED]
    collided = states[..., COLLIDED]
    d = states[..., DISTANCE]
    dpad = torch.clamp(pad_d[:, None] - d, -GAP_CAP, GAP_CAP)
    dgate = torch.clamp(gate_d[:, None] - d, -GAP_CAP, GAP_CAP)
    return torch.stack([x, heading, speed, collided, dpad, dgate], dim=-1)


def compute_stats(feats: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = feats.reshape(-1, FEATURE_DIM)
    deltas = (states[:, 1:, :4] - states[:, :-1, :4]).reshape(-1, 4)
    return {
        "feat_mean": flat.mean(0),
        "feat_std": flat.std(0).clamp(min=1e-4),
        "delta_mean": deltas.mean(0),
        "delta_std": deltas.std(0).clamp(min=1e-4),
    }


def tokens(feats: torch.Tensor, actions: torch.Tensor, stats: dict) -> torch.Tensor:
    """(B, T, F) feats + (B, T) actions -> (B, T, INPUT_DIM)."""
    z = (feats - stats["feat_mean"]) / stats["feat_std"]
    onehot = torch.nn.functional.one_hot(actions.long(), N_ACTIONS).to(z.dtype)
    return torch.cat([z, onehot], dim=-1)
