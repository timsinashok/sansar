"""Tests for the V0.5 gate environment and probes.

The load-bearing ones are the benchmark-validity tests: exact label balance,
a clean gate readout, and no contamination of the step the readout is taken
from. A memory benchmark that leaks is worse than no benchmark, so these are
asserted rather than eyeballed.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sansar.core.types import Action, CarState
from sansar.envs.driving.physics import ClassicEngine
from sansar.envs.gate.physics import GateEngine
from sansar.envs.gate.vec import collect_trials
from sansar.envs.gate.world import BIT, COLLIDED, DISTANCE, HEADING, SPEED, X
from sansar.models.sequence import build


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"), OmegaConf.load("configs/gate.yaml")
    )


def test_base_dynamics_match_v0_classic_engine(cfg):
    """With the gate far away, gate physics must equal V0 physics step for step.

    Guards against V0.5 results being explained by an accidentally different
    vehicle model rather than by the memory mechanic.
    """
    v0 = cfg.copy()
    v0.env.obstacles.enabled = False
    v0.env.road.kind = "straight"
    v0.env.car.initial_speed = cfg.env.car.initial_speed
    classic = ClassicEngine(v0.env)
    gate = GateEngine(cfg.env, pad_distance=1e9, gate_distance=1e9)

    cs = classic.reset()
    gs = gate.reset()
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = int(rng.integers(0, 3))
        cs = classic.step(cs, Action(a))
        gs = gate.step(gs, a)
        assert gs[X] == pytest.approx(cs.x, abs=1e-9)
        assert gs[HEADING] == pytest.approx(cs.heading, abs=1e-9)
        assert gs[SPEED] == pytest.approx(cs.speed, abs=1e-9)
        assert gs[DISTANCE] == pytest.approx(cs.distance, abs=1e-9)
        assert bool(gs[COLLIDED]) == cs.collided


def test_labels_exactly_balanced(cfg):
    """Chance must be exactly 0.5, not 0.5 +- sampling noise, or a
    majority-class predictor beats 'chance'."""
    for gap in (64, 256):
        d = collect_trials(cfg.env, 400, gap, seed=3)
        assert d["gate_open"].mean() == pytest.approx(0.5, abs=1e-12)


def test_gate_readout_is_clean_and_uncontaminated(cfg):
    """Closed gates collide at t_hit, open gates do not, and NOTHING has
    collided at t_hit - 1 (which is the last step the model may look at)."""
    d = collect_trials(cfg.env, 600, 256, seed=5)
    n = len(d["t_hit"])
    ar = np.arange(n)
    st, th, op = d["states"], d["t_hit"], d["gate_open"]
    assert st[ar, th, COLLIDED][~op].mean() == pytest.approx(1.0)
    assert st[ar, th, COLLIDED][op].mean() == pytest.approx(0.0)
    assert st[ar, th - 1, COLLIDED].mean() == pytest.approx(0.0)


def test_pad_side_sets_the_bit(cfg):
    """The hidden bit is a function of which side of the pad line we crossed."""
    d = collect_trials(cfg.env, 400, 128, seed=7)
    ar = np.arange(len(d["t_hit"]))
    x_at_pad = d["states"][ar, d["t_pad"], X]
    bit = d["states"][ar, d["t_hit"], BIT]
    assert np.all((x_at_pad < 0) == (bit > 0.5))


def test_process_noise_destroys_the_trajectory_hash(cfg):
    """With deterministic dynamics the float value of x is a hash of the whole
    trajectory and the bit is recoverable from one step. Process noise must
    flatten that: no fine-grained bin of x may predict the label."""
    d = collect_trials(cfg.env, 3000, 256, seed=11)
    ar = np.arange(3000)
    x1 = d["states"][ar, d["t_hit"] - 1, X]
    edges = np.percentile(x1, np.linspace(0, 100, 13))
    idx = np.clip(np.digitize(x1, edges[1:-1]), 0, 11)
    for b in range(12):
        m = idx == b
        if m.sum() >= 50:
            assert abs(d["gate_open"][m].mean() - 0.5) < 0.12, f"bin {b} leaks"


@pytest.mark.parametrize("name", ["memoryless", "mlp", "transformer", "gru"])
def test_models_share_one_interface(cfg, name):
    """predict_at and the incremental rollout interface work for every
    architecture — the probes are written once and applied to all of them."""
    m = build(name, cfg.gate_model)
    B, T = 4, 60
    tok = torch.randn(B, T, 9)
    out = m.predict_at(tok, torch.full((B,), 40))
    assert out.shape == (B, 5)
    mem = m.init_mem(B, "cpu")
    o, mem = m.step_token(tok[:, 0], mem)
    assert o.shape == (B, 5)


def test_collection_is_reproducible_from_seed(cfg):
    a = collect_trials(cfg.env, 64, 128, seed=21)
    b = collect_trials(cfg.env, 64, 128, seed=21)
    assert np.array_equal(a["states"], b["states"])
    assert np.array_equal(a["actions"], b["actions"])
