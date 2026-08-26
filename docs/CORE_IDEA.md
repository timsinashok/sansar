# Sansar — Core Idea

> **A transformer as a neural game engine.**
>
> Instead of the game world existing as hand-written code with a neural network
> merely controlling an agent inside it, the neural network itself becomes the
> learned dynamical system that the game world runs on.

---

## 1. The one-line thesis

Build a small playable game whose world simulation is performed by a
transformer rather than by traditional physics/game logic. The transformer
maintains a **latent representation of the game world** and predicts how that
world evolves in response to player actions.

```
Player input
     │
     ▼
┌─────────────────┐
│   Transformer   │
│                 │
│ latent world    │
│ state + action  │
└────────┬────────┘
         │
         ▼
   next latent state
         │
         ▼
      Decoder
         │
         ▼
   game state / screen
```

The long-term question:

> **Can a transformer act as a persistent, interactive simulator of a game
> world?** Not just predict images, and not just control an existing game —
> actually *run* the underlying world dynamics.

---

## 2. Game engine vs. world model — the key distinction

This project is closely related to world-model research, which learns

```
z_{t+1} = f(z_t, a_t)
```

where `z_t` is a latent world state and `a_t` an action. Sansar does
essentially this — but takes it one step further:

> **The learned world model *becomes* the game simulator.**

| Conventional | Sansar |
|---|---|
| `Player → hand-written engine → next state` | `Player → transformer → next state` |

The renderer is then just a way of *visualizing* the latent simulation.

### Contrast with DOOM-GPT

In DOOM-GPT, the actual DOOM simulator (ViZDoom) still runs the game; the
transformer's hidden activations are used to reconstruct frames. The
transformer does **not** simulate the game logic. In Sansar, the transformer
itself is the simulator:

```
player action → transformer → next latent world → renderer
```

---

## 3. V0: the smallest possible neural game

Deliberately tiny. Not GTA, not 3D, not even a complicated driving sim.

**A very small 2D driving game:**

```
┌─────────────────────┐
│                     │
│        ROAD         │
│                     │
│        🚗           │
│                     │
│                     │
└─────────────────────┘

       ◀       ▶
```

### V0 features
- 2D top-down / forward-scrolling road
- One player car
- Left/right steering + forward movement
- Simple road boundaries and simple collision
- Optional score/distance counter
- Fixed simulation timestep
- Basic renderer

**No** NPCs, complex physics, missions, or large maps. The point is the
smallest environment that can demonstrate the neural-engine concept — and the
player must actually be able to play it.

### What we explicitly avoid in V0
GTA-scale environments · 3D graphics · pixel prediction · huge transformers ·
complex physics · many NPCs · RL training from scratch · procedurally
generated massive worlds · solving everything at once.

The first objective is simply:

> Make a tiny game where the traditional simulator generates data, train a
> latent transformer on that data, **replace the simulator with the
> transformer**, and actually play the resulting neural game.

---

## 4. Training strategy: the conventional simulator as data generator

No existing game is needed. We write a tiny conventional simulator ourselves
with ordinary equations (position, velocity, heading, steering):

```
Player input → simple deterministic physics → game state → renderer
```

While playing (or auto-generating trajectories), we record transitions:

```
(state_t, action_t, state_{t+1})

state₀ + left  → state₁
state₁ + left  → state₂
state₂ + right → state₃
...
```

The conventional simulator is thus the **data generator and ground truth**.

---

## 5. Latent architecture

We do not ask the transformer to predict pixels. We use a latent bottleneck:

```
Game state ─▶ Encoder ─▶ z_t ─(+ action)─▶ Transformer ─▶ z_{t+1} ─▶ Decoder ─▶ predicted game state
```

Eventually the latent chain runs autoregressively without decoding/re-encoding
every step — the latent state itself becomes the persistent simulation state:

```
z₀ → z₁ → z₂ → z₃ → z₄ → ...
```

### Why latent space instead of pixels?

`image → transformer → next image` conflates two hard problems:
1. understanding/simulating the world
2. generating realistic pixels

By keeping structured game state → latent dynamics → structured state →
conventional renderer, we isolate the actual research question:

> **Can a transformer learn the dynamics of an interactive world?**

Rendering stays completely conventional.

---

## 6. Simulation frequency ≠ rendering frequency (50 Hz / 10 Hz)

Run simulation at 50 Hz (Δt = 20 ms), render at ~10 Hz (Δt = 100 ms):

```
sim:    z0 → z1 → z2 → z3 → z4 → z5 → ... → z10 → ...
render: z0                        z5         z10
```

This decouples **simulation frequency** from **visualization frequency**, and
keeps the neural simulator from being tied to rendering. The render rate can
increase later; the separation is the important idea.

---

## 7. Memory — the central research theme

A world model isn't just next-step prediction; it must maintain a **coherent
world over time**. For a world model, memory is more fundamental than for an
RL agent:

- RL agent memory: *"I saw an enemy 10 seconds ago."*
- World-model memory: *"That enemy still exists even though it isn't
  currently visible."* — the **world itself** needs persistent state.

### The canonical example

```
Gate = OPEN
```

The player opens a gate, drives away (gate leaves observation), drives around
for thirty seconds, and returns. The model should still produce
`Gate = OPEN` — not treat the world as reset.

### Latent state as compressed world memory

Ideally `z_t` becomes a sufficient compressed statistic of the world:

```
z_t
├── player position / velocity
├── road geometry
├── object positions & states
├── NPC states
├── previous events
├── persistent world changes
└── hidden information needed for future prediction
```

No explicit variables required — the research question is whether the
transformer learns a useful latent representation such that
`P(z_{t+1} | z_t, a_t)` contains enough information to simulate the future.

### The gate experiment (flagship memory test)

1. Player encounters a gate 🚧
2. Player opens it → `OPEN`
3. Player drives away; gate leaves the observation
4. Player drives around for a long time
5. Player returns → the neural world must still say `🚧 = OPEN`

This tests genuine persistent latent world state — considerably more
interesting than testing whether the car moves correctly.

---

## 8. Long-horizon stability > one-step accuracy

The biggest known failure mode of learned world models: recursive rollout
drift. A model may predict the next state very accurately, yet when fed its
own predictions —

```
ground truth:  s0 → s1 → s2 → ... → s1000
neural:        s0 → ŝ1 → ŝ2 → ... → ŝ1000
```

— small errors compound until the simulated world becomes nonsense.

**Core experiment:** measure divergence between ground-truth and neural
rollouts at 1, 10, 50, 100, 500, 1 000 steps (and eventually much longer).
Long-horizon coherent simulation is the interesting result, not one-step MSE.

---

## 9. The baseline: two versions of the same game

We maintain both engines side by side. The conventional simulator generates
training data and serves as ground truth; then we swap it out.

```
Traditional:  player input → hand-written physics → state → renderer
Neural:       player input → transformer → latent state → decoder → renderer
```

| Property | Traditional | Neural |
|---|---|---|
| Position | Ground truth | Predicted |
| Velocity | Ground truth | Predicted |
| Collision | Ground truth | Predicted |
| Persistent state | Exact | Learned |
| Long rollout | Stable | **Test** |
| Human playable | Yes | **Target** |

This makes every comparison clean and direct.

---

## 10. Research questions (measurable)

The framing is **not** "can a transformer drive a car?" It is:

> Can a transformer maintain a compact latent representation of a persistent
> interactive world and autoregressively simulate that world over long
> horizons — well enough that a human can play inside it?

1. **Representation** — does `z_t` capture useful world information?
2. **Dynamics** — can it predict how the world changes?
3. **Memory** — does it preserve information no longer observable?
4. **Long-horizon stability** — thousands of steps without collapse?
5. **Controllability** — do different actions yield correct counterfactual
   futures?
6. **Interactivity** — can a human actually play the neural simulation?

---

## 11. Roadmap

```
V0    2D car + straight road + simple physics + transformer latent dynamics
V0.1  curved roads
V0.2  obstacles
V0.3  NPC cars
V0.4  persistent objects        ← gate experiment
V0.5  NPC memory
V1    small 3D driving world
…     larger worlds, more entities, persistent events, complex NPC behavior
```

### Later extensions (explicitly deferred)

- **NPCs with persistent state** — an NPC that remembers the player, so
  something the player did 30 seconds ago influences later events. At that
  point we're learning persistent world dynamics, not just physics.
- **Multi-timescale transformers** — e.g. 50 Hz low-level vehicle dynamics,
  10 Hz local world dynamics, 2 Hz high-level events/NPC decisions. V0 uses a
  single transformer; hierarchy only if performance demands it.
- **AntMaze** — a reasonable benchmark (long-horizon, navigation, memory,
  planning, continuous actions) but rejected as V0 because it reframes the
  project as "world model for an RL agent." The game framing — *a game whose
  world is simulated by a transformer* — is the motivation. AntMaze can serve
  as a validation environment later.
- **The GTA-scale question** — the extreme endpoint: can a sufficiently large
  transformer maintain and evolve a persistent, interactive open world
  (vehicles, NPCs, traffic, weather, missions, world history) via
  `z_{t+1} = Transformer(z_t, a_t)`? Not a starting point; a direction.

---

## 12. The final vision

```
        SIMPLE GAME
             │
    collect trajectories
             │
    learn latent dynamics
             │
  autoregressive transformer
             │
   persistent latent world
             │
     playable neural game
             │
    add memory + entities
             │
      larger environments
             │
        3D game worlds
             │
    potentially open worlds
```

---

## 13. Open questions to validate externally

Three things to actively challenge:

1. Is the latent-state formulation technically sound?
2. What existing world-model literature already does something very similar
   (e.g. Dreamer-family latent world models, GameNGen-style neural game
   engines, Genie-style playable world models)?
3. What would make this **scientifically interesting** rather than just "a
   transformer predicting game states"? (Current best answer: persistent
   memory of unobserved state + long-horizon playable stability.)
