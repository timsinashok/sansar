"""Collect training trajectories from scripted policies.

    python scripts/collect.py [key=value ...]

Examples:
    python scripts/collect.py
    python scripts/collect.py data.episodes=500 data.run_name=v0_big
    python scripts/collect.py env.road.kind=curved data.run_name=v0_curved
"""

import sys

from sansar.data.collect import collect
from sansar.utils.config import load_config


def main() -> None:
    collect(load_config(sys.argv[1:]))


if __name__ == "__main__":
    main()
