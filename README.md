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

📄 Full write-up: [docs/CORE_IDEA.md](docs/CORE_IDEA.md)
