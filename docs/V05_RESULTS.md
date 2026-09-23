# V0.5 results — the gate memory experiment

Generated from `experiments/v05/results.json`. Figure: `experiments/v05/v05_results.png`.

### Stage 1 — leak curve (validity of the benchmark)

A window model provably cannot know the bit once the pad is outside its
context, so any score above chance there is leakage, not memory.

| realised gap | memoryless (K=1) | transformer (K=32) | verdict |
|---|---|---|---|
| 9 | 1.000 | 1.000 | pad inside window — n/a |
| 26 | 1.000 | 1.000 | pad inside window — n/a |
| 58 | 0.648 | 1.000 | **LEAKS — excluded** |
| 122 | 0.500 | 0.546 | clean |
| 250 | 0.528 | 0.520 | clean |
| 506 | 0.502 | 0.466 | clean |
| 1018 | 0.500 | 0.532 | clean |

### Stage 2 — memory sweep (the claim)

Gate-prediction accuracy, mean ± sd over 3 seeds. Chance is exactly 0.500.

| realised gap | memoryless | mlp | transformer | gru |
|---|---|---|---|---|
| 122 | 0.519 ± 0.015 | 0.528 ± 0.021 | 0.537 ± 0.007 | 1.000 ± 0.000 |
| 250 | 0.513 ± 0.011 | 0.507 ± 0.014 | 0.517 ± 0.006 | 1.000 ± 0.000 |
| 506 | 0.503 ± 0.002 | 0.504 ± 0.007 | 0.495 ± 0.020 | 0.999 ± 0.001 |
| 1018 | 0.502 ± 0.003 | 0.513 ± 0.008 | 0.518 ± 0.011 | 0.846 ± 0.218 |

Parameter counts: memoryless 270k, mlp 676k, transformer 799k, gru 925k.

### Stage 3 — context control (H1)

Gap held fixed at 250 steps; only the transformer's window K varies.
H1 predicts chance for K < gap and ~1.0 for K > gap.

| K | gate accuracy | vs gap |
|---|---|---|
| 8 | 0.529 ± 0.017 | K < gap |
| 32 | 0.517 ± 0.006 | K < gap |
| 64 | 0.525 ± 0.015 | K < gap |
| 128 | 0.518 ± 0.023 | K < gap |
| 256 | 1.000 ± 0.000 | K > gap |
| 320 | 1.000 ± 0.000 | K > gap |

### Stage 4 — stability (probe 2)

Free rollout. RMSE is not reported: on a bounded state space it rewards
mean-collapse. Time-to-failure and Wasserstein distance do not.

| model | TTF median | TTF p10 | survived full horizon | W1(x) | W1(speed) |
|---|---|---|---|---|---|
| memoryless | 96 | 37 | 0% | 0.672 | 1.006 |
| mlp | 36 | 11 | 0% | 0.462 | 1.924 |
| transformer | 240 | 37 | 0% | 0.794 | 1.346 |
| gru | 354 | 262 | 0% | 0.591 | 0.835 |

---

## Reading of the results

**H1 (positive control) — passed.** Holding the gap fixed at 250 steps and
varying only the transformer's window gives a clean step function: chance at
K = 8/32/64/128, exactly 1.000 at K = 256/320. The cliff falls precisely where
the pad line enters the window. The harness measures what it claims to.

**H2 (the claim) — supported.** The recurrent latent reaches 1.000 at gaps of
122, 250 and 506 steps, where every window model is at chance and where the
bit is provably absent from a 32-step window. Parameter counts are comparable
(gru 925k vs transformer 799k), so this is an architecture result, not a
capacity result.

**H3 (graceful degradation) — falsified.** At a 1018-step gap the three seeds
scored 1.000, 1.000 and 0.538. Retention does not decay smoothly; it either
works or collapses to chance. The ~1000-step limit is therefore an
optimisation-reliability failure, not a capacity ceiling — a distinction that
is only visible because the protocol runs 3 seeds. A single-seed run would
have reported "solved" or "fails" purely on luck.

**Memory and stability are uncorrelated.** The transformer has the second-best
time-to-failure (240 steps) and chance-level memory; the mlp has the worst TTF
(36) and chance-level memory; the gru leads both (354 / 1.000). Had these been
one conflated metric, as in V0, neither effect would have been visible.

**Nothing is stable under free rollout.** No architecture survived 1000
open-loop steps — best median TTF is 354 (gru). V0's headline claimed a
"neural engine" holding 1000 steps; under a metric that mean-collapse cannot
game, none of these models does.

## Limitations — stated, not buried

1. **The clean regime starts at a 122-step gap.** Below that the pad-side
   excursion is still decaying inside the window (the leak curve shows the
   transformer at 1.000 for a 58-step gap). Short-horizon memory is untestable
   in this design; testing it needs a mechanic that does not require lateral
   displacement.
2. **The retained quantity is one bit.** This is persistence of a single
   hidden fact, not of a world state. Nothing here shows a model maintaining
   multiple objects, relations, or a scene.
3. **Process noise is load-bearing.** The benchmark is only valid because the
   dynamics are stochastic. Any deterministic world model benchmark built on
   this codebase will leak through trajectory-hash channels.
4. **Teacher-forced memory probe.** Probe 1 feeds ground-truth history by
   design, to separate retention from drift. It therefore says nothing about
   whether the bit survives inside a *self-generated* rollout — and given that
   no model survives 1000 free-running steps, it very likely does not.
5. **One environment, one mechanic, one gap per model.** Each cell trains on a
   single gap. Generalisation across gaps is untested.
6. **The GRU is not the thesis architecture.** It is a recurrent baseline that
   establishes the gap is closable. Whether a transformer variant with
   recurrence or state-space memory does as well is open.
