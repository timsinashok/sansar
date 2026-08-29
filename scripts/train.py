"""Train the dynamics transformer on a collected dataset.

    python scripts/train.py [key=value ...]

Examples:
    python scripts/train.py
    python scripts/train.py train.steps=10000 train.run_name=m3_long
    python scripts/train.py train.data_run=v0_curved runtime.device=cpu
"""

import sys

from sansar.training.trainer import train
from sansar.utils.config import load_config


def main() -> None:
    train(load_config(sys.argv[1:]))


if __name__ == "__main__":
    main()
