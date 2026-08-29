import numpy as np
import torch

from sansar.data.collect import collect
from sansar.data.dataset import EpisodeData
from sansar.eval.rollout import baseline_rollout, divergence_metrics, rollout
from sansar.models.dynamics import DynamicsTransformer
from sansar.models.features import FEATURE_DIM, Geometry, compute_stats, tokens
from sansar.training.trainer import train
from sansar.utils.config import load_config

TINY = [
    "runtime.device=cpu",
    "model.context=8",
    "model.d_model=32",
    "model.n_layers=1",
    "model.n_heads=2",
    "train.batch_size=16",
    "train.steps=12",
    "train.eval_every=12",
    "train.eval_horizon=10",
    "train.eval_windows=16",
    "train.val_episodes=1",
    "train.rollout_curriculum=[[0,1],[6,2]]",
]


def test_geometry_features():
    cfg = load_config()
    geom = Geometry(cfg.env, "cpu", max_distance=2000.0)
    states = torch.tensor([[0.5, 0.1, 10.0, 100.0, 0.0]])
    f = geom.features(states)
    assert f.shape == (1, FEATURE_DIM)
    # straight road: x_rel == x, curvature lookaheads are zero
    assert f[0, 0] == 0.5
    assert f[0, 8] == 0.0 and f[0, 9] == 0.0
    # blocker gap is positive and capped
    assert 0.0 < f[0, 4] <= 30.0


def test_model_forward_shape():
    cfg = load_config(["model.context=8", "model.d_model=32", "model.n_layers=1", "model.n_heads=2"])
    model = DynamicsTransformer(cfg.model)
    out = model(torch.zeros(3, 8, 13))
    assert out.shape == (3, 5)


def test_rollout_and_baseline_shapes(tmp_path):
    cfg = load_config([f"data.out_dir={tmp_path}", "data.episodes=2", "data.steps=100", *TINY])
    collect(cfg)
    data = EpisodeData.load(tmp_path / cfg.data.run_name)
    geom = Geometry(cfg.env, "cpu", max_distance=2000.0)
    stats = compute_stats(geom, torch.from_numpy(data.flat_states()))
    model = DynamicsTransformer(cfg.model)
    rng = np.random.default_rng(0)
    states, actions = data.sample_windows(4, 8, 10, rng, "cpu")
    toks = tokens(geom, stats, states[:, :8], actions[:, :8])
    assert toks.shape == (4, 8, 13)
    preds, loss = rollout(model, geom, stats, states, actions, 8, 10, compute_loss=True)
    assert preds.shape == (4, 10, 5)
    assert torch.isfinite(loss)
    base = baseline_rollout(states, 8, 10)
    m = divergence_metrics(base, states[:, 8:])
    assert all(np.isfinite(v) for v in m.values())


def test_train_smoke_writes_checkpoint(tmp_path):
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
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    assert "state_dict" in ckpt and "stats" in ckpt
    assert np.isfinite(ckpt["metrics"]["model/x_rmse_H"])
