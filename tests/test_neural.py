import numpy as np

from sansar.core.types import Action, CarState
from sansar.data.collect import collect
from sansar.neural.engine import NeuralEngine
from sansar.training.trainer import train
from sansar.utils.config import load_config
from tests.test_model import TINY


def test_neural_engine_runs_stably(tmp_path):
    cfg = load_config(
        [
            f"data.out_dir={tmp_path}",
            f"experiment.out_dir={tmp_path}/exp",
            "data.episodes=3",
            "data.steps=150",
            *TINY,
        ]
    )
    collect(cfg)
    run_dir = train(cfg)

    eng = NeuralEngine(run_dir / "model.pt", device="cpu")
    s = eng.reset()
    assert isinstance(s, CarState)
    rng = np.random.default_rng(0)
    for _ in range(30):
        s = eng.step(s, Action(rng.integers(0, 3)))
    arr = s.to_array()
    assert np.isfinite(arr).all()
    # a barely-trained model is inaccurate, but must stay in a sane envelope
    assert abs(s.x) < 20 and 0 <= s.speed < 50


def test_neural_engine_reset_clears_history(tmp_path):
    cfg = load_config(
        [
            f"data.out_dir={tmp_path}",
            f"experiment.out_dir={tmp_path}/exp",
            "data.episodes=2",
            "data.steps=120",
            *TINY,
        ]
    )
    collect(cfg)
    run_dir = train(cfg)
    eng = NeuralEngine(run_dir / "model.pt", device="cpu")

    s = eng.reset()
    first = [eng.step(s, Action.RIGHT).to_array()]
    for _ in range(5):
        first.append(eng.step(CarState.from_array(first[-1]), Action.RIGHT).to_array())

    s = eng.reset()
    second = [eng.step(s, Action.RIGHT).to_array()]
    for _ in range(5):
        second.append(eng.step(CarState.from_array(second[-1]), Action.RIGHT).to_array())

    np.testing.assert_array_equal(np.stack(first), np.stack(second))
