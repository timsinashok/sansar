"""Play the driving game.

    python scripts/play.py [key=value ...]

Examples:
    python scripts/play.py                          # classic hand-written engine
    python scripts/play.py game.engine=neural       # the trained transformer
    python scripts/play.py game.mode=duel           # neural car + classic ghost, same inputs
    python scripts/play.py env.road.kind=curved

Controls: arrows / A-D to steer, R to reset, Esc to quit.
"""

import sys

from sansar.game.loop import run, run_duel
from sansar.utils.config import load_config


def main() -> None:
    cfg = load_config(sys.argv[1:])

    if cfg.game.mode == "duel":
        from sansar.envs.driving.physics import ClassicEngine
        from sansar.neural.engine import NeuralEngine

        neural = NeuralEngine(cfg.game.checkpoint, device=str(cfg.game.neural_device))
        cfg.env = neural.env_cfg  # both engines and renderer share the trained world
        run_duel(cfg, neural, ClassicEngine(cfg.env))
        return

    if cfg.game.engine == "neural":
        from sansar.neural.engine import NeuralEngine

        engine = NeuralEngine(cfg.game.checkpoint, device=str(cfg.game.neural_device))
        cfg.env = engine.env_cfg  # render the world the checkpoint was trained on
        title = "sansar — NEURAL engine (transformer)"
    else:
        from sansar.envs.driving.physics import ClassicEngine

        engine = ClassicEngine(cfg.env)
        title = "sansar — classic engine"
    run(cfg, engine, title)


if __name__ == "__main__":
    main()
