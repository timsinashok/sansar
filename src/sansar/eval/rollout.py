"""Autoregressive rollout, the trivial baseline, and divergence metrics.

Used by both training (pushforward rollout loss) and evaluation (open-loop
divergence vs ground truth) — one implementation, so the number train
optimizes is the number eval reports.
"""

import torch
from torch.nn import functional as F

from sansar.models.dynamics import DynamicsTransformer
from sansar.models.features import Geometry, integrate, tokens


def rollout(
    model: DynamicsTransformer,
    geom: Geometry,
    stats: dict,
    states: torch.Tensor,      # (B, context+horizon, 5) ground truth
    actions: torch.Tensor,     # (B, context+horizon-1)
    context: int,
    horizon: int,
    compute_loss: bool = False,
    collision_weight: float = 1.0,
):
    """Roll the model on its own predictions for `horizon` steps.

    With compute_loss=True this is pushforward training: per-step losses
    against ground truth, with the fed-back state detached so each step gets
    a clean one-step gradient on model-generated input distributions.
    Returns (predicted absolute states (B, horizon, 5), mean loss or None).
    """
    window = states[:, :context]
    preds, losses = [], []
    for j in range(horizon):
        toks = tokens(geom, stats, window, actions[:, j : j + context])
        out = model(toks)
        last = window[:, -1]

        if compute_loss:
            target = states[:, context + j]
            target_delta = (target[:, :4] - last[:, :4] - stats["delta_mean"]) / stats[
                "delta_std"
            ]
            loss = F.huber_loss(out[:, :4], target_delta) + collision_weight * (
                F.binary_cross_entropy_with_logits(out[:, 4], target[:, 4])
            )
            losses.append(loss)

        nxt = integrate(last, out, stats)
        preds.append(nxt)
        window = torch.cat([window[:, 1:], nxt.detach().unsqueeze(1)], dim=1)

    loss = torch.stack(losses).mean() if losses else None
    return torch.stack(preds, dim=1), loss


def baseline_rollout(states: torch.Tensor, context: int, horizon: int) -> torch.Tensor:
    """Constant-velocity (delta-repeat) baseline: keep applying the last
    observed one-step delta. The model must beat this to mean anything."""
    last, prev = states[:, context - 1], states[:, context - 2]
    delta = last[:, :4] - prev[:, :4]
    steps = torch.arange(1, horizon + 1, device=states.device, dtype=states.dtype)
    pred4 = last[:, None, :4] + delta[:, None, :] * steps[None, :, None]
    collided = last[:, 4].unsqueeze(1).expand(-1, horizon).unsqueeze(-1)
    return torch.cat([pred4, collided], dim=-1)


def divergence_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """preds/targets (B, H, 5) -> RMSE of lateral position at step 1 and at
    the final step, distance RMSE at the final step, collision accuracy."""
    err = preds[..., :4] - targets[..., :4]
    return {
        "x_rmse_1": err[:, 0, 0].pow(2).mean().sqrt().item(),
        "x_rmse_H": err[:, -1, 0].pow(2).mean().sqrt().item(),
        "dist_rmse_H": err[:, -1, 3].pow(2).mean().sqrt().item(),
        "coll_acc": (preds[..., 4] == targets[..., 4]).float().mean().item(),
    }
