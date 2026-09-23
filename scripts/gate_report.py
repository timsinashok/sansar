"""Turn experiments/v05/results.json into the V0.5 results table and figure."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def agg(rows, keys):
    """Group rows by `keys` and aggregate gate_acc across seeds."""
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    res = {}
    for k, rs in out.items():
        a = np.array([r["gate_acc"] for r in rs])
        res[k] = {
            "mean": a.mean(), "std": a.std(), "n": len(a),
            "realised_gap": rs[0]["realised_gap"],
            "ci_lo": min(r["ci_lo"] for r in rs), "ci_hi": max(r["ci_hi"] for r in rs),
            "n_params": rs[0]["n_params"],
        }
    return res


def markdown(data: dict) -> str:
    L = []
    W = "\n"
    if "leak" in data:
        L.append("### Stage 1 — leak curve (validity of the benchmark)\n")
        L.append("A window model provably cannot know the bit once the pad is outside its")
        L.append("context, so any score above chance there is leakage, not memory.\n")
        d = agg(data["leak"], ["model", "gap"])
        gaps = sorted({k[1] for k in d})
        L.append("| realised gap | memoryless (K=1) | transformer (K=32) | verdict |")
        L.append("|---|---|---|---|")
        for g in gaps:
            m = d.get(("memoryless", g)); t = d.get(("transformer", g))
            rg = (m or t)["realised_gap"]
            if rg <= 32:
                v = "pad inside window — n/a"
            elif t and t["mean"] > 0.60:
                v = "**LEAKS — excluded**"
            else:
                v = "clean"
            L.append(f"| {rg} | {m['mean']:.3f} | {t['mean']:.3f} | {v} |")
        L.append("")
    if "memory" in data:
        L.append("### Stage 2 — memory sweep (the claim)\n")
        L.append("Gate-prediction accuracy, mean ± sd over 3 seeds. Chance is exactly 0.500.\n")
        d = agg(data["memory"], ["model", "gap"])
        gaps = sorted({k[1] for k in d}); models = [m for m in ["memoryless","mlp","transformer","gru"] if any(k[0]==m for k in d)]
        L.append("| realised gap | " + " | ".join(models) + " |")
        L.append("|---" * (len(models) + 1) + "|")
        for g in gaps:
            rg = next(v["realised_gap"] for k, v in d.items() if k[1] == g)
            cells = []
            for m in models:
                v = d.get((m, g))
                cells.append(f"{v['mean']:.3f} ± {v['std']:.3f}" if v else "—")
            L.append(f"| {rg} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Parameter counts: " + ", ".join(
            f"{m} {next(v['n_params'] for k,v in d.items() if k[0]==m)/1e3:.0f}k" for m in models) + ".")
        L.append("")
    if "context" in data:
        L.append("### Stage 3 — context control (H1)\n")
        d = agg(data["context"], ["context"])
        rg = next(iter(d.values()))["realised_gap"]
        L.append(f"Gap held fixed at {rg} steps; only the transformer's window K varies.")
        L.append("H1 predicts chance for K < gap and ~1.0 for K > gap.\n")
        L.append("| K | gate accuracy | vs gap |")
        L.append("|---|---|---|")
        for k in sorted(d):
            v = d[k]
            L.append(f"| {k[0]} | {v['mean']:.3f} ± {v['std']:.3f} | {'K < gap' if k[0] < rg else 'K > gap'} |")
        L.append("")
    if "stability" in data:
        L.append("### Stage 4 — stability (probe 2)\n")
        L.append("Free rollout. RMSE is not reported: on a bounded state space it rewards")
        L.append("mean-collapse. Time-to-failure and Wasserstein distance do not.\n")
        L.append("| model | TTF median | TTF p10 | survived full horizon | W1(x) | W1(speed) |")
        L.append("|---|---|---|---|---|---|")
        for r in data["stability"]:
            L.append(f"| {r['model']} | {r['ttf_median']:.0f} | {r['ttf_p10']:.0f} | "
                     f"{r['ttf_frac_survived']:.0%} | {r['w1_x']:.3f} | {r['w1_speed']:.3f} |")
        L.append("")
    return W.join(L)


def figure(data: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"memoryless": "#8c8b88", "mlp": "#d68a2a", "transformer": "#2a78d6", "gru": "#1f9d55"}
    TEXT, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e7e6e3", "#fcfcfb"
    panels = [k for k in ("leak", "memory", "context") if k in data]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), facecolor=BG)
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
        if key == "context":
            d = agg(data[key], ["context"])
            ks = sorted(k[0] for k in d)
            rg = next(iter(d.values()))["realised_gap"]
            ax.errorbar(ks, [d[(k,)]["mean"] for k in ks], yerr=[d[(k,)]["std"] for k in ks],
                        color=COL["transformer"], marker="o", ms=5, lw=2, capsize=3)
            ax.axvline(rg, color="#c0392b", ls="--", lw=1.4)
            ax.annotate(f"gap = {rg}", (rg, 0.56), color="#c0392b", fontsize=9,
                        ha="right", rotation=90, va="bottom")
            ax.set_xscale("log")
            ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
            ax.set_xlabel("transformer context K (steps)", fontsize=9, color=MUTED)
            ax.set_title("H1 control: window vs a fixed 250-step gap", fontsize=11, color=TEXT, loc="left")
        else:
            d = agg(data[key], ["model", "gap"])
            for m in ["memoryless", "mlp", "transformer", "gru"]:
                pts = sorted(((v["realised_gap"], v) for k, v in d.items() if k[0] == m))
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1]["mean"] for p in pts]
                es = [p[1]["std"] for p in pts]
                ax.errorbar(xs, ys, yerr=es, color=COL[m], marker="o", ms=5, lw=2,
                            capsize=3, label=m)
            ax.set_xscale("log")
            allx = sorted({v["realised_gap"] for v in d.values()})
            ax.set_xticks(allx)
            ax.set_xticklabels([str(v) for v in allx])
            ax.minorticks_off()
            ax.set_xlabel("realised pad→gate gap (steps)", fontsize=9, color=MUTED)
            if key == "leak":
                ax.axvspan(1, 32, color="#c0392b", alpha=0.07)
                ax.set_title("Leak curve: where the benchmark is valid", fontsize=11, color=TEXT, loc="left")
            else:
                ax.set_title("Memory: gate recall vs gap", fontsize=11, color=TEXT, loc="left")
            ax.legend(fontsize=8, frameon=False, loc="center right")
        ax.set_ylim(0.35, 1.05)
        ax.set_ylabel("gate prediction accuracy", fontsize=9, color=MUTED)
        ax.grid(True, which="both", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.98, 0.512, "chance", transform=ax.get_yaxis_transform(), fontsize=8,
                color=MUTED, ha="right")

    fig.suptitle("Sansar V0.5 — the gate memory experiment", fontsize=14, color=TEXT,
                 x=0.01, y=0.99, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/v05")
    ap.add_argument("--out", default="docs/V05_RESULTS.md")
    a = ap.parse_args()
    data = json.loads((Path(a.run) / "results.json").read_text())
    fig_path = figure(data, Path(a.run) / "v05_results.png")
    md = markdown(data)
    Path(a.out).write_text(
        "# V0.5 results — the gate memory experiment\n\n"
        f"Generated from `{a.run}/results.json`. Figure: `{fig_path}`.\n\n" + md
    )
    print(md)
    print(f"\nwrote {a.out} and {fig_path}")
