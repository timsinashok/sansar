# V0 Plan

V0 is four deliverables in sequence, the last reusing everything from the
first:

```
Phase 1: classic game        Phase 2: data           Phase 3: training         Phase 4: neural game
─────────────────────        ──────────────          ─────────────────         ────────────────────
pygame loop                  scripted drivers +      transformer learns        same pygame loop,
50 Hz physics step()         human play sessions     s_{t+1} from history      step() = model.predict()
render at 10 Hz              → (s, a, s') logs       + rollout losses          → you play the model
```

The load-bearing trick: the game loop, data collection, and evals are all
written against one interface (`sansar.core.engine.Engine`) —
`step(state, action) -> next_state`. The neural engine is a drop-in
replacement, which also enables a side-by-side mode (ghost car = ground
truth, solid car = neural, identical inputs).

## Repo organization (built to scale past V0)

```
sansar/
  configs/                 # committed YAML configs; local.yaml (gitignored) per machine
  scripts/                 # thin CLIs: play.py, collect.py, train.py, eval.py
  src/sansar/
    core/                  # Engine interface, CarState/Action — env-agnostic contracts
    envs/driving/          # V0 env: road geometry + classic physics
    render/                # pygame renderer (reads state only, engine-agnostic)
    game/                  # interactive loop (sim_hz stepping, render_hz drawing)
    data/                  # collection policies, episode logging, datasets   [Phase 2]
    models/                # transformer dynamics models                       [Phase 3]
    training/              # trainer, rollout losses, schedules               [Phase 3]
    eval/                  # divergence curves, event accuracy, side-by-side  [Phase 4]
    utils/                 # config loading, device resolution
  experiments/             # run outputs: checkpoints, logs, config snapshots (gitignored)
  data_out/                # collected episodes (gitignored)
  tests/
  docs/
```

Principles:

1. **Envs are pluggable.** Every future game version (curved roads,
   obstacles, NPCs, persistent objects) is an env behind the same `Engine`
   interface. V0.1 (curved roads) is already just
   `env.road.kind=curved` in config — zero new code.
2. **Everything is config-driven.** No hyperparameters or physics constants
   in code.
3. **Thin scripts, fat library.** Scripts parse overrides and call into
   `src/sansar/`.

## Configuration conventions

Precedence (later wins):

```
configs/default.yaml   <   configs/local.yaml   <   CLI key=value overrides
(committed defaults)       (gitignored,             python scripts/play.py \
                            machine-specific:         env.render_hz=50
                            device, paths)
```

- `runtime.device: auto` resolves cuda > mps > cpu at runtime; pin it per
  machine in `configs/local.yaml` (copy `local.example.yaml`).
- Every training/eval run snapshots its fully-resolved config into
  `experiments/<run>/config.yaml` via `sansar.utils.config.save_config` —
  reproducibility by default.
- `.env` is reserved for secrets only (e.g. a future W&B key), never for
  hyperparameters or device selection.

## Phase 1 — classic game  ✅ done

- Fixed-timestep physics at `env.sim_hz` (50), rendering every Nth state at
  `env.render_hz` (10) — the sim/visualization frequency split from the core
  idea. Set `env.render_hz=50` for smooth play.
- State: `(x, heading, speed, distance, collided)`; actions: NONE/LEFT/RIGHT.
- Collision = clamp to road edge + slow down (not episode end): keeps
  episodes continuous and generates rich boundary data for training.
- Road centerline is a function of distance → curved roads are free.
- Run it: `python scripts/play.py` (arrows/A-D steer, R reset, Esc quit).

## Phase 2 — data collection

Log `(state, action)` sequences per episode to `data_out/` as `.npz`.
Coverage matters more than volume — mix of scripted policies:

- random steering (state-space coverage)
- noisy lane-following (typical play)
- deliberate wall-hitters (collision events are rare; oversample them)
- swervy/adversarial inputs (what a player will do to break the model)
- genuine human play via the game loop

At 50 Hz the simulator runs far faster than real time; 1–10 M transitions is
minutes of compute.

## Phase 3 — the transformer

Two architectures, in order:

- **A (V0): context-as-memory.** GPT-style over a window of past
  `(state, action)` tokens → next state. Memory = attention over history.
  Easy to train; gets us a playable neural game fast.
- **B (V0.5): recurrent latent.** A single `z_t` carried forward —
  the thesis formulation. The gate/memory experiments only become meaningful
  here, once information must survive outside the observation window.

Training choices:

- Predict **normalized state deltas** (`Δs`), not absolute states.
- Continuous regression (Huber/MSE); no state tokenization in V0.
- **Rollout losses** — the key to long-horizon stability: unroll the model on
  its own predictions for k steps (curriculum k: 1 → 5 → 20 → 50) and
  backprop through the rollout. Pure teacher forcing collapses at long
  horizons.
- Separate **collision head** (BCE): collision is a discontinuous event and
  the hardest thing for a smooth regressor to capture.
- Tiny model: ~4 layers, 128–256 dim. Must forward-pass in ≪20 ms on MPS/CPU
  to hold 50 Hz — trivial at this size.
- Always compare against the trivial baseline: constant-velocity
  extrapolation.

## Phase 4 — neural game + evals

1. **Divergence curves** (headline figure): matched initial state + identical
   action sequence, plot state error at 1/10/50/100/500/1000 steps.
2. **Event accuracy:** collisions called within ± a few frames.
3. **Playability:** does steering feel right; does the world survive minutes
   of play.
4. **Side-by-side demo:** ghost = classic, solid = neural, same inputs.

## Milestones

- [x] **M1** — playable classic game behind the `Engine` interface
- [x] **M2** — data pipeline; 600 k transitions from mixed policies
      (`scripts/collect.py`, 12% collision steps, replay-exact determinism)
- [x] **M3** — transformer beats constant-velocity baseline at 1-step and
      50-step on lateral position (x@1: 2.4 mm vs 6.6 mm; x@50: 0.11 m vs
      1.63 m — 12×) with 97% collision accuracy. Caveat: 50-step *distance*
      RMSE (3–4 m) is still worse than the baseline (1.2 m) — speed is
      near-constant so delta-repeat is nearly exact there; improving
      speed/distance drift is open iteration.
- [x] **M4** — playable neural game: `play.py game.engine=neural`, 1.35 ms
      inference/step (15× under the 20 ms budget), 40 s erratic-input
      rollout stays finite and on-road (subjective play-feel: user to judge)
- [x] **M5** — divergence report (`scripts/eval.py` → per-horizon RMSE table,
      JSON, and figure in the checkpoint dir) + duel mode
      (`play.py game.mode=duel`: neural solid car, classic ground-truth ghost,
      identical inputs, live drift HUD). Headline numbers at 1000 steps
      (20 s open-loop): lateral 0.75 m vs 30.8 m baseline (41×), heading
      0.043 vs 38 rad, 93.6% collision accuracy; distance crosses below the
      baseline beyond ~200 steps (22.0 m vs 32.7 m at 1000) but remains the
      weakest axis at short horizons.

**V0 is complete** — the loop from the core idea is closed: classic game →
trajectories → latent dynamics transformer → playable neural game, with
quantified long-horizon divergence. Next frontiers (V0.5): recurrent-latent
architecture (architecture B) and the persistent-state gate experiment.

## Decisions taken (revisit if needed)

| Decision | Choice | Why |
|---|---|---|
| Architecture order | context-window first, recurrent latent as V0.5 | playable artifact fast; latent is where the memory research lives |
| Collision semantics | clamp + slow, episodes continue | far more boundary data than end-on-crash |
| Road | straight by default, curved via config | smallest-possible V0, but one flag makes the dynamics nonlinear |
| Config system | OmegaConf YAML + local.yaml + CLI overrides | Hydra-like power without the weight |
