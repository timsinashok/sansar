import numpy as np

from sansar.core.types import CarState
from sansar.data.collect import collect, replay
from sansar.envs.driving.physics import ClassicEngine
from sansar.utils.config import load_config


def small_cfg(tmp_path, *overrides):
    return load_config(
        [
            f"data.out_dir={tmp_path}",
            "data.episodes=4",
            "data.steps=200",
            *overrides,
        ]
    )


def test_collect_writes_episodes_with_expected_shapes(tmp_path):
    cfg = small_cfg(tmp_path)
    out = collect(cfg)
    files = sorted(out.glob("ep_*.npz"))
    assert len(files) == 4
    assert (out / "config.yaml").exists()
    ep = np.load(files[0])
    assert ep["states"].shape == (201, CarState.DIM)
    assert ep["actions"].shape == (200,)
    assert str(ep["policy"]) in {"lane", "random", "swerve", "wall"}


def test_episodes_replay_exactly(tmp_path):
    cfg = small_cfg(tmp_path)
    out = collect(cfg)
    engine = ClassicEngine(cfg.env)
    for f in sorted(out.glob("ep_*.npz")):
        ep = np.load(f)
        resim = replay(ep["states"], ep["actions"], engine)
        np.testing.assert_array_equal(resim, ep["states"])


def test_same_seed_reproduces_dataset(tmp_path):
    a = collect(small_cfg(tmp_path / "a"))
    b = collect(small_cfg(tmp_path / "b"))
    for fa, fb in zip(sorted(a.glob("ep_*.npz")), sorted(b.glob("ep_*.npz"))):
        ea, eb = np.load(fa), np.load(fb)
        np.testing.assert_array_equal(ea["states"], eb["states"])
        np.testing.assert_array_equal(ea["actions"], eb["actions"])


def test_dataset_covers_collisions_and_all_actions(tmp_path):
    cfg = small_cfg(tmp_path, "data.episodes=8", "data.steps=500")
    out = collect(cfg)
    states = np.concatenate([np.load(f)["states"] for f in out.glob("ep_*.npz")])
    actions = np.concatenate([np.load(f)["actions"] for f in out.glob("ep_*.npz")])
    assert states[:, 4].sum() > 0, "no collision steps in dataset"
    assert set(np.unique(actions)) == {0, 1, 2}, "some action never taken"
