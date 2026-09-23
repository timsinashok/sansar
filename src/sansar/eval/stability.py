"""Probe 2 — long-horizon stability, measured without RMSE.

V0 reported RMSE against ground truth. On a bounded state space that metric
*rewards* the failure it should detect: a model that collapses to driving
down the middle scores 1.559 while `std(x)` is 1.558, i.e. indistinguishable
from a good model by that number alone. Two metrics that mean-collapse cannot
game:

  time-to-failure     steps until the rollout leaves the training support
                      (off-road, speed out of range, or non-finite). Reported
                      as a distribution — median and 10th percentile.

  distribution match  1-D Wasserstein distance between the marginals of x and
                      speed under long neural rollout vs the classic engine.
                      A collapsed model has a narrower marginal and fails this
                      while still winning on RMSE.
"""

import numpy as np
import torch

from sansar.envs.gate.world import DISTANCE, SPEED, X
from sansar.models.gate_features import features, tokens


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance via matched quantiles."""
    q = np.linspace(0.0, 1.0, 512)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


@torch.no_grad()
def free_rollout(model, ts, horizon: int, n: int, device: str) -> torch.Tensor:
    """Roll the model on its OWN predictions for `horizon` steps.

    Seeded with the model's context worth of ground-truth history, then closed
    loop: every subsequent input feature is recomputed from the model's own
    predicted state.
    """
    model.eval()
    n = min(n, len(ts))
    warm = max(model.context, 1)
    state = ts.states[:n, warm].clone()
    mem = model.init_mem(n, device)
    stats = {k: v.to(device) for k, v in ts.stats.items()}
    pad, gate = ts.pad_d[:n], ts.gate_d[:n]

    for t in range(warm):
        mem = model.step_token(ts.tokens[:n, t], mem)[1]

    out_states = torch.empty(n, horizon, state.shape[-1], device=device)
    for j in range(horizon):
        f = features(state[:, None], pad, gate)
        a = ts.actions[:n, (warm + j) % ts.actions.shape[1]]
        tok = tokens(f, a[:, None], stats)[:, 0]
        pred, mem = model.step_token(tok, mem)
        delta = pred[:, :4] * stats["delta_std"] + stats["delta_mean"]
        nxt = state.clone()
        nxt[:, :4] = state[:, :4] + delta
        nxt[:, 4] = (pred[:, 4] > 0).to(state.dtype)
        state = nxt
        out_states[:, j] = state
    model.train()
    return out_states


def time_to_failure(states: torch.Tensor, road_half_width: float, car_half_width: float,
                    speed_lo: float, speed_hi: float) -> np.ndarray:
    """First step at which a rollout leaves the training support (else horizon)."""
    x, sp = states[..., X], states[..., SPEED]
    limit = road_half_width - car_half_width + 0.05
    bad = (
        ~torch.isfinite(states).all(-1)
        | (x.abs() > limit)
        | (sp < speed_lo - 0.5)
        | (sp > speed_hi + 0.5)
    )
    H = bad.shape[1]
    any_bad = bad.any(1)
    first = torch.where(any_bad, bad.float().argmax(1), torch.full_like(any_bad, H, dtype=torch.long))
    return first.cpu().numpy()


def stability_report(model, ts, cfg, device: str, horizon: int = 1000, n: int = 256) -> dict:
    roll = free_rollout(model, ts, horizon, n, device)
    car, road = cfg.env.car, cfg.env.road
    ttf = time_to_failure(
        roll, float(road.half_width), float(car.half_width),
        float(car.collision_min_speed), float(car.max_speed),
    )
    true_states = ts.states[:n, 1 : horizon + 1]
    m = min(true_states.shape[1], horizon)
    alive = torch.from_numpy(ttf).to(device)[:, None] > torch.arange(m, device=device)[None, :]
    rx = roll[:, :m, X][alive].cpu().numpy()
    rs = roll[:, :m, SPEED][alive].cpu().numpy()
    tx = true_states[..., X].reshape(-1).cpu().numpy()
    tsp = true_states[..., SPEED].reshape(-1).cpu().numpy()
    return {
        "ttf_median": float(np.median(ttf)),
        "ttf_p10": float(np.percentile(ttf, 10)),
        "ttf_frac_survived": float((ttf >= horizon).mean()),
        "w1_x": wasserstein1(rx, tx) if len(rx) else float("nan"),
        "w1_speed": wasserstein1(rs, tsp) if len(rs) else float("nan"),
        "horizon": horizon,
    }
