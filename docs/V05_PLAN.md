# V0.5 Plan — the memory experiment

V0 closed the loop (classic game → trajectories → transformer → playable
neural game) but could not test the project's central claim. This plan fixes
that. It is a redesign of the *experiment*, not just an extension of the
model.

---

## 1. Why V0's evaluation did not support its conclusion

Four defects, all confirmed against the M5 artifacts:

**The baseline was not a baseline.** M5's headline was "lateral 0.75 m vs
30.8 m — 41×". The road is 5.2 m wide (`x ∈ [-2.6, 2.6]`). A baseline
reporting 30.79 m of lateral error is predicting the car six road-widths
outside the world. Measured against baselines that stay inside the physical
world, on the same 128 held-out windows:

| horizon | model | freeze | center (x≡0) | const-velocity |
|---|---|---|---|---|
| 50   | 0.058 | 0.775 | 1.590 | 1.52 |
| 1000 | **0.752** | **1.749** | 1.559 | 30.79 |

The true margin is **2.3×**, not 41×. On speed it is 1.6× (1.809 vs 2.885).

**The state space is bounded, so RMSE cannot detect mean-collapse.**
`std(x) = 1.558`, and "always predict dead center" scores 1.559. A model that
collapses to driving down the middle is rewarded by this metric.

**The collision metric was cumulative and unbalanced.**
`(preds[:, :h, 4] == targets[:, :h, 4]).mean()` averages over steps `1..h`,
so "93.6% at 1000 steps" is really the mean over the first 1000 steps,
dominated by early accuracy. Collisions are 14% of validation steps, so
always-predict-no-collision scores **86.0%**.

**Checkpoint selection ran on the flattered axis.** `trainer.py` saves on
`model/x_rmse_H` alone — lateral RMSE, the metric a bounded state space makes
easy — while speed and distance, the axes that actually fail, are ignored.

Underneath all four: the V0 task is a smooth deterministic ODE with six of
thirteen input dimensions supplied by a ground-truth world oracle. It cannot
distinguish a world model from a curve fitter, and it has no memory content
at all — so no result from it bears on the research question.

## 2. The design principle

> Build a task whose trivial-model floor is **provable**, not empirical.

Everything else follows.

---

## 3. The gate environment

One mechanic added to the V0 driving env:

```
   ┌─────────────────────────┐
   │ ///// PAD LINE \\\\\\\\ │  ← the side you cross on sets a hidden bit
   │   left = OPEN           │
   │            right = CLOSED│
   │                         │
   │          ...            │  ← G steps of ordinary driving
   │      (funnel to center) │     pad far outside the 30 m sensor cap
   │                         │
   │  ═══════ GATE ═══════   │  ← spans the road; blocks iff bit = CLOSED
   └─────────────────────────┘
```

- Crossing the **pad line** at `x < centerline` sets `gate = OPEN`;
  at `x >= centerline` sets `gate = CLOSED`. No new action; the bit is set by
  *where you were*, which is what makes it implicit.
- `G` steps later the **gate** spans the full road. If CLOSED the car
  collides and is slowed hard; if OPEN it passes through untouched.
- **The readout is one bit:** does the model's collision head fire at the
  gate? No new model machinery, no new metric.

**Why a memoryless model provably scores 50%.** The gate's state is not a
function of any observable at gate time. Trials are balanced 50/50 by
construction. Hence chance = 0.5 exactly, and this is a *floor*, not an
empirical baseline that might be a strawman.

### Anti-leak measures

The obvious way this benchmark breaks is `x` at gate time still correlating
with the pad taken. Three mitigations:

1. **Policy funnel.** Every scripted driver steers to the centerline for the
   final `funnel_m` metres before the gate, so gate-time `x ≈ 0` regardless
   of pad.
2. **Blockers disabled** in the gate env, so the two paths cannot differ in
   speed via differing obstacle encounters.
3. **The leak check is run first and gates the whole experiment** (§6).

### Observation features

`FEATURE_DIM` 10 → 12; two features are added and **neither carries the
bit**:

| feature | meaning |
|---|---|
| `dpad` | forward gap to the pad line (capped, signed until crossed) |
| `dgate` | forward gap to the gate (capped) |

The bit is computable from `(dpad ≈ 0, sign(x_rel))` at a single instant and
is required at `dgate ≈ 0`, `G` steps later. That gap is the whole experiment.

---

## 4. What gets measured

Memory and stability are **separate probes**. V0 conflated them; a model can
fail the memory test purely from rollout drift, which tells you nothing about
memory.

### Probe 1 — memory (teacher-forced)

Inputs are ground-truth observation history, so there is **no compounding-error
confound**. Read the collision logit at `t_gate` (first step past the gate
distance). Report accuracy with a Wilson 95% interval against the 0.5 floor.
Sweep `G ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 2048}`.

### Probe 2 — stability (free rollout)

RMSE is replaced by two metrics that mean-collapse cannot game:

1. **Time-to-failure** — steps until the trajectory leaves training support
   (`x` off-road, speed outside `[3, 13.9]`, non-finite). Report the
   *distribution*: median and 10th percentile.
2. **Distributional match** — 1-D Wasserstein distance between the marginals
   of `x` and `speed` under long neural rollout vs the classic engine. A
   mean-collapsed model has a narrower marginal and fails this while winning
   on RMSE.

---

## 5. The control ladder

Matched parameter budget across every row. Each answers exactly one question.

| Model | Question |
|---|---|
| `memoryless` (context = 1) | Does the task leak? **Must be ≈ 0.5.** |
| `mlp` (flattened K-window) | Does *attention* matter, or just history? |
| `transformer` K ∈ {1, 8, 32, 128} | Where is the cliff, and does it track K? |
| `gru` (carried latent, no window) | Can recurrence beat a window at equal params? |

## 6. Pre-registered hypotheses

Written before running. Two are falsifiable predictions about *our own*
model, one is a control.

- **H1 (positive control).** The context-`K` transformer scores ≈ 1.0 for
  `G < K` and ≈ 0.5 for `G > K`, transitioning at `G = K`.
  *If H1 fails, the harness is broken — not the model.* Nothing else in this
  document may be believed until H1 passes.
- **H2 (the claim).** The GRU scores above chance for `G >> K` at matched
  parameter count.
- **H3 (the risky one).** The GRU degrades *gracefully* with `G` rather than
  cliff-edged.

H1-as-control is what makes this an experiment rather than a demo: the design
yields a publishable negative result even if H2 fails.

### The leak check runs first

Train the `memoryless` model and confirm ≈ 0.5. If it exceeds the CI on 0.5,
the task leaks: strengthen the funnel and re-run. **No other result is
reported until the leak check passes.**

## 7. Statistical protocol

- **3 seeds per condition**, mean ± std. One run is an anecdote.
- **≥ 400 held-out trials** per (model, G) cell.
- **Held-out worlds, not just trajectories**: train on pad/gate placements
  drawn from one seed set, evaluate on unseen ones.
- **Checkpoint selection on gate accuracy**, never on a single regression axis.

## 8. Layout

```
src/sansar/envs/gate/      world.py (pad+gate geometry), physics.py, policies.py
src/sansar/models/         sequence.py (unified step interface), gate_features.py
src/sansar/training/       gate_trainer.py
src/sansar/eval/           memory.py (probe 1 + leak check), stability.py (probe 2)
scripts/                   gate_collect.py, gate_train.py, gate_eval.py
configs/gate.yaml
```

All models implement one protocol so the probes are written once:

```python
init_memory(batch) -> mem
step(token, mem)   -> (out, mem)
```

Transformer/MLP carry a token window as `mem`; the GRU carries a hidden state.
