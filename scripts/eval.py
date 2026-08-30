"""Run the divergence report against the checkpoint in game.checkpoint.

    python scripts/eval.py [key=value ...]

Examples:
    python scripts/eval.py
    python scripts/eval.py game.checkpoint=experiments/m3/model.pt
"""

import sys

from sansar.eval.divergence import DIMS, HORIZONS, run_divergence
from sansar.utils.config import load_config


def main() -> None:
    result, fig = run_divergence(load_config(sys.argv[1:]))
    print(f"\n{'horizon':>8} | " + " | ".join(f"{d + ' (m/b)':>22}" for d in DIMS) + " | coll acc")
    for i, h in enumerate(HORIZONS):
        row = " | ".join(
            f"{result['model'][d][i]:>10.4f}/{result['baseline'][d][i]:<11.4f}" for d in DIMS
        )
        print(f"{h:>8} | {row} | {result['model']['coll_acc'][i]:>7.3f}")
    print(f"\nfigure: {fig}")


if __name__ == "__main__":
    main()
