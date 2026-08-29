"""Training loop for the dynamics transformer.

Pushforward rollout training: each batch rolls the model on its own
(detached) predictions for k steps, with k growing on a curriculum, so the
model learns to correct the compounding error that pure teacher forcing
never exposes. Validation reports open-loop divergence vs the
constant-velocity baseline — the M3 criterion.
"""

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sansar.data.dataset import EpisodeData
from sansar.eval.rollout import baseline_rollout, divergence_metrics, rollout
from sansar.models.dynamics import DynamicsTransformer
from sansar.models.features import Geometry, compute_stats
from sansar.utils.config import resolve_device, save_config


def _curriculum_k(curriculum, step: int) -> int:
    k = 1
    for start, val in curriculum:
        if step >= int(start):
            k = int(val)
    return k


@torch.no_grad()
def evaluate(model, geom, stats, val: EpisodeData, cfg, rng, device) -> dict:
    context = int(cfg.model.context)
    horizon = int(cfg.train.eval_horizon)
    states, actions = val.sample_windows(
        int(cfg.train.eval_windows), context, horizon, rng, device
    )
    targets = states[:, context:]
    model.eval()
    preds, _ = rollout(model, geom, stats, states, actions, context, horizon)
    model.train()
    m = {f"model/{k}": v for k, v in divergence_metrics(preds, targets).items()}
    base = baseline_rollout(states, context, horizon)
    m.update({f"base/{k}": v for k, v in divergence_metrics(base, targets).items()})
    return m


def train(cfg: DictConfig) -> Path:
    device = resolve_device(cfg.runtime.device)
    torch.manual_seed(int(cfg.runtime.seed))
    rng = np.random.default_rng(int(cfg.runtime.seed))

    run_dir = Path(cfg.experiment.out_dir) / cfg.train.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    data = EpisodeData.load(Path(cfg.data.out_dir) / cfg.train.data_run)
    train_data, val_data = data.split(int(cfg.train.val_episodes))
    max_d = float(data.states[..., 3].max()) + 500.0
    geom = Geometry(cfg.env, device, max_distance=max_d)
    stats = compute_stats(geom, torch.from_numpy(train_data.flat_states()).to(device))

    model = DynamicsTransformer(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device} params={n_params/1e6:.2f}M episodes={train_data.states.shape[0]}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    context = int(cfg.model.context)
    steps = int(cfg.train.steps)
    curriculum = [list(x) for x in cfg.train.rollout_curriculum]
    best = float("inf")
    t0 = time.time()

    for step in range(1, steps + 1):
        k = _curriculum_k(curriculum, step)
        states, actions = train_data.sample_windows(
            int(cfg.train.batch_size), context, k, rng, device
        )
        _, loss = rollout(
            model, geom, stats, states, actions, context, k,
            compute_loss=True,
            collision_weight=float(cfg.train.collision_loss_weight),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % int(cfg.train.eval_every) == 0 or step == steps:
            m = evaluate(model, geom, stats, val_data, cfg, rng, device)
            h = int(cfg.train.eval_horizon)
            print(
                f"[{step:5d}] k={k} loss={loss.item():.4f} | "
                f"x@1 {m['model/x_rmse_1']:.4f}m (base {m['base/x_rmse_1']:.4f}) | "
                f"x@{h} {m['model/x_rmse_H']:.3f}m (base {m['base/x_rmse_H']:.3f}) | "
                f"dist@{h} {m['model/dist_rmse_H']:.3f}m (base {m['base/dist_rmse_H']:.3f}) | "
                f"coll {m['model/coll_acc']:.3f} | {time.time() - t0:.0f}s"
            )
            if m["model/x_rmse_H"] < best:
                best = m["model/x_rmse_H"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "stats": {k_: v.cpu() for k_, v in stats.items()},
                        "model_cfg": OmegaConf.to_container(cfg.model),
                        "env_cfg": OmegaConf.to_container(cfg.env),
                        "metrics": m,
                        "step": step,
                    },
                    run_dir / "model.pt",
                )

    print(f"[train] done in {time.time() - t0:.0f}s — best x@{cfg.train.eval_horizon} = {best:.3f}m")
    print(f"[train] checkpoint: {run_dir / 'model.pt'}")
    return run_dir
