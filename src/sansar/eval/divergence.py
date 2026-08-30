"""The M5 divergence report: how far does the neural simulation drift from
ground truth as it runs open-loop?

From matched initial states and identical recorded action sequences, the
model rolls out 1000 steps (20 s of sim time) on its own predictions; RMSE
against the classic engine is reported at each horizon, alongside the
constant-velocity baseline. Produces a JSON of metrics and the headline
figure.
"""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sansar.data.dataset import EpisodeData
from sansar.eval.rollout import baseline_rollout, rollout
from sansar.models.dynamics import DynamicsTransformer
from sansar.models.features import Geometry
from sansar.utils.config import resolve_device

HORIZONS = [1, 10, 50, 100, 200, 500, 1000]
DIMS = ["x", "heading", "speed", "distance"]


def _per_horizon(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """preds/targets (B, H, 5) -> {dim: [rmse per horizon], coll_acc: [...]}."""
    err = preds[..., :4] - targets[..., :4]
    out = {d: [] for d in DIMS}
    out["coll_acc"] = []
    for h in HORIZONS:
        for i, d in enumerate(DIMS):
            out[d].append(err[:, h - 1, i].pow(2).mean().sqrt().item())
        out["coll_acc"].append(
            (preds[:, :h, 4] == targets[:, :h, 4]).float().mean().item()
        )
    return out


@torch.no_grad()
def run_divergence(cfg: DictConfig, n_windows: int = 128) -> tuple[dict, Path]:
    device = resolve_device(cfg.runtime.device)
    ckpt_path = Path(cfg.game.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = DynamicsTransformer(OmegaConf.create(ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
    context = int(ckpt["model_cfg"]["context"])

    data = EpisodeData.load(Path(cfg.data.out_dir) / cfg.train.data_run)
    _, val = data.split(int(cfg.train.val_episodes))
    geom = Geometry(
        OmegaConf.create(ckpt["env_cfg"]), device,
        max_distance=float(data.states[..., 3].max()) + 500.0,
    )

    horizon = max(HORIZONS)
    rng = np.random.default_rng(int(cfg.runtime.seed))
    states, actions = val.sample_windows(n_windows, context, horizon, rng, device)
    targets = states[:, context:]

    print(f"[divergence] rolling {n_windows} windows x {horizon} steps on {device}...")
    preds, _ = rollout(model, geom, stats, states, actions, context, horizon)
    base = baseline_rollout(states, context, horizon)

    result = {
        "checkpoint": str(ckpt_path),
        "horizons": HORIZONS,
        "n_windows": n_windows,
        "model": _per_horizon(preds, targets),
        "baseline": _per_horizon(base, targets),
    }
    out_json = ckpt_path.parent / "divergence.json"
    out_json.write_text(json.dumps(result, indent=2))
    fig_path = _plot(result, ckpt_path.parent / "divergence.png")
    return result, fig_path


def _plot(result: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MODEL, BASE = "#2a78d6", "#52514e"
    TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e3"
    h = result["horizons"]
    units = {"x": "m", "heading": "rad", "speed": "m/s", "distance": "m"}
    titles = {
        "x": "Lateral position",
        "heading": "Heading",
        "speed": "Speed",
        "distance": "Distance traveled",
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), facecolor="#fcfcfb")
    for ax, dim in zip(axes.flat, DIMS):
        ax.set_facecolor("#fcfcfb")
        ax.plot(h, result["model"][dim], color=MODEL, lw=2, marker="o", ms=4, label="Neural engine")
        ax.plot(h, result["baseline"][dim], color=BASE, lw=2, ls="--", marker="o", ms=4,
                label="Constant-velocity baseline")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(titles[dim], fontsize=11, color=TEXT, loc="left")
        ax.set_ylabel(f"RMSE ({units[dim]})", fontsize=9, color=MUTED)
        ax.set_xlabel("rollout horizon (steps, 50 Hz)", fontsize=9, color=MUTED)
        ax.grid(True, which="both", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].legend(fontsize=9, frameon=False, loc="upper left")

    coll = result["model"]["coll_acc"][-1]
    fig.suptitle("Neural engine divergence from ground truth", fontsize=14,
                 color=TEXT, x=0.02, y=0.99, ha="left", fontweight="bold")
    fig.text(0.02, 0.935,
             f"Open-loop rollout, identical actions, {result['n_windows']} held-out windows "
             f"· collision accuracy over 1000 steps: {coll:.1%}",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
