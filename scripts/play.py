"""Play the driving game with the classic engine.

    python scripts/play.py [key=value ...]

Examples:
    python scripts/play.py
    python scripts/play.py env.road.kind=curved
    python scripts/play.py env.render_hz=50

Controls: arrows / A-D to steer, R to reset, Esc to quit.
"""

import sys

from sansar.envs.driving.physics import ClassicEngine
from sansar.game.loop import run
from sansar.utils.config import load_config


def main() -> None:
    cfg = load_config(sys.argv[1:])
    run(cfg, ClassicEngine(cfg.env))


if __name__ == "__main__":
    main()
