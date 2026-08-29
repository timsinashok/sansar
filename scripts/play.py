"""Play the driving game.

    python scripts/play.py [key=value ...]

Examples:
    python scripts/play.py                          # classic hand-written engine
    python scripts/play.py game.engine=neural       # the trained transformer
    python scripts/play.py game.engine=neural game.checkpoint=experiments/m3_long/model.pt
    python scripts/play.py env.road.kind=curved

Controls: arrows / A-D to steer, R to reset, Esc to quit.
"""

import sys

from sansar.game.loop import run
from sansar.utils.config import load_config


def main() -> None:
    cfg = load_config(sys.argv[1:])
    if cfg.game.engine == "neural":
        from sansar.neural.engine import NeuralEngine

        engine = NeuralEngine(cfg.game.checkpoint, device=str(cfg.game.neural_device))
        # render the world the checkpoint was trained on
        cfg.env = engine.env_cfg
        title = "sansar — NEURAL engine (transformer)"
    else:
        from sansar.envs.driving.physics import ClassicEngine

        engine = ClassicEngine(cfg.env)
        title = "sansar — classic engine"
    run(cfg, engine, title)


if __name__ == "__main__":
    main()
