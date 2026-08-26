# sansar

**A transformer as a neural game engine.**

Sansar explores whether a transformer can act as a persistent, interactive
simulator of a game world — not predicting images, not controlling an agent
inside an existing game, but actually *running* the underlying world dynamics
as a latent autoregressive process:

```
z_{t+1} = Transformer(z_t, a_t)
```

V0 is a tiny playable 2D driving game with two interchangeable engines: a
hand-written simulator (data generator + ground truth) and a learned latent
transformer that replaces it. The central research themes are **persistent
world memory** (does a gate you opened stay open after it leaves the screen?)
and **long-horizon rollout stability** (thousands of steps without collapse).

📄 Vision: [docs/CORE_IDEA.md](docs/CORE_IDEA.md) · Plan & repo layout: [docs/V0_PLAN.md](docs/V0_PLAN.md)

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # add ".[train]" for torch when training
cp configs/local.example.yaml configs/local.yaml   # set your device etc.
.venv/bin/python scripts/play.py    # arrows/A-D steer, R reset, Esc quit
```

Config precedence: `configs/default.yaml` < `configs/local.yaml` (gitignored,
machine-specific) < CLI overrides, e.g.
`python scripts/play.py env.road.kind=curved env.render_hz=50`.
