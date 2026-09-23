"""V0.5 experiment runner: leak curve, memory sweep, context control, stability.

    python scripts/gate_run.py                       # everything
    python scripts/gate_run.py --only leak           # one stage
    python scripts/gate_run.py --out experiments/v05

Every stage writes incrementally to <out>/results.json so a long run can be
inspected (or resumed) while it is still going.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sansar.training.gate_trainer import TrialSet, train_one
from sansar.utils.config import resolve_device


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest CI for a proportion near 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _record(path: Path, stage: str, row: dict):
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(stage, []).append(row)
    path.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/v05")
    ap.add_argument("--only", default="all",
                    choices=["all", "leak", "memory", "context", "stability"])
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"), OmegaConf.load("configs/gate.yaml")
    )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    device = resolve_device(cfg.runtime.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    res = out / "results.json"
    t_start = time.time()
    print(f"[v05] device={device} out={out}")

    def run(name, gap, seed, ctx=None):
        c = cfg.copy()
        if ctx is not None:
            c.gate_model.context = ctx
        r = train_one(name, gap, seed, c, device)
        r["context"] = ctx if ctx is not None else int(cfg.gate_model.context)
        k = round(r["gate_acc"] * r["n_eval"])
        r["ci_lo"], r["ci_hi"] = wilson(k, r["n_eval"])
        return r

    # ---- Stage 1: leak curve -------------------------------------------------
    # Beyond its context a window model provably cannot know the bit, so any
    # score above chance there measures residual leakage. This defines the
    # valid regime; nothing downstream is believable outside it.
    if args.only in ("all", "leak"):
        print("\n== stage 1: leak curve ==")
        for gap in cfg.sweep.leak_gaps:
            for name in ["memoryless", "transformer"]:
                r = run(name, gap, 0)
                r["stage"] = "leak"
                _record(res, "leak", r)
                print(f"  {name:12s} G={gap:5d} realised={r['realised_gap']:5d} "
                      f"acc={r['gate_acc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")

    # ---- Stage 2: memory sweep (the claim) -----------------------------------
    if args.only in ("all", "memory"):
        print("\n== stage 2: memory sweep ==")
        for gap in cfg.sweep.gaps:
            for name in cfg.sweep.models:
                for seed in cfg.sweep.seeds:
                    r = run(name, gap, seed)
                    r["stage"] = "memory"
                    _record(res, "memory", r)
                    print(f"  {name:12s} G={gap:5d} s={seed} acc={r['gate_acc']:.3f} "
                          f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  ({time.time()-t_start:.0f}s)")

    # ---- Stage 3: context control (H1) ---------------------------------------
    # H1 tested the sound way round: hold the gap fixed and clean, vary the
    # model's window. Accuracy must jump from chance to ~1.0 as K crosses the
    # realised gap. If it does not, the harness is broken.
    if args.only in ("all", "context"):
        print("\n== stage 3: context control (H1) ==")
        gap = int(cfg.sweep.context_gap)
        for ctx in cfg.sweep.contexts:
            for seed in cfg.sweep.seeds:
                r = run("transformer", gap, seed, ctx=ctx)
                r["stage"] = "context"
                _record(res, "context", r)
                print(f"  K={ctx:5d} G={gap} s={seed} realised={r['realised_gap']} "
                      f"acc={r['gate_acc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")

    # ---- Stage 4: stability (probe 2) ----------------------------------------
    if args.only in ("all", "stability"):
        print("\n== stage 4: stability ==")
        from sansar.eval.stability import stability_report

        gap = int(cfg.sweep.context_gap)
        for name in cfg.sweep.models:
            for seed in cfg.sweep.seeds[:1]:
                m, model, ev = train_one(name, gap, seed, cfg, device, return_model=True)
                rep = stability_report(model, ev, cfg, device, horizon=int(cfg.sweep.stability_horizon))
                rep.update({"model": name, "seed": seed, "gap": gap, "gate_acc": m["gate_acc"]})
                _record(res, "stability", rep)
                print(f"  {name:12s} ttf_med={rep['ttf_median']:.0f} p10={rep['ttf_p10']:.0f} "
                      f"survived={rep['ttf_frac_survived']:.2f} W1x={rep['w1_x']:.3f} W1spd={rep['w1_speed']:.3f}")

    print(f"\n[v05] done in {time.time()-t_start:.0f}s -> {res}")


if __name__ == "__main__":
    main()
