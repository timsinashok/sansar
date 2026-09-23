"""Vectorised trial generation for the gate env.

All N episodes of a run share the same target gap G, so they run in lockstep
as numpy arrays — the whole sweep collects in seconds rather than minutes.

The scripted driver has four phases, and phase C+D exist purely as anti-leak
measures (V0_5_PLAN §3):

    A wander       random lanes + action noise (general dynamics coverage)
    B pad approach steer to the chosen side of the pad line
    C centre       steer to the centreline and stay there until the gate

Phase C is deliberately the most boring option available. Two richer designs
were tried and both leaked:

  * funnel only near the gate — the car was still drifting in from the pad
    side inside the window, and a K=32 transformer scored 0.94 at gap 58;
  * an independent "decoy" lane after the pad — the TRANSIT from pad lane to
    decoy lane was itself visible, and the same model scored 0.995 at gap 59.

Returning to centre immediately confines the observable trace to the ~15-step
return transit right after the pad, so any gap beyond context + ~15 steps is
clean. Verified empirically per gap by eval/memory.py, not assumed.

The leak detector is not the memoryless model (one step cannot see a transit
direction) but the window transformer itself: at realised gaps beyond its
context the bit is provably absent from its input, so any score above 0.5 is
residual leakage, quantified.
"""

import numpy as np
from omegaconf import DictConfig

from sansar.envs.gate.physics import CarParams, step_batch
from sansar.envs.gate.world import BIT, COLLIDED, DISTANCE, HEADING, SPEED, STATE_DIM, X

NONE, LEFT, RIGHT = 0, 1, 2


def _steer_toward(x, heading, target_x):
    """Vectorised bang-bang controller (mirrors data/policies._steer_toward)."""
    desired = np.clip(0.5 * (target_x - x), -0.4, 0.4)
    return np.where(desired > heading + 0.02, RIGHT, np.where(desired < heading - 0.02, LEFT, NONE))


def collect_trials(
    env_cfg: DictConfig,
    n: int,
    target_gap_steps: int,
    seed: int,
) -> dict:
    """Simulate n gate trials with a pad->gate gap of ~target_gap_steps."""
    p = CarParams.from_cfg(env_cfg)
    g = env_cfg.gate
    rng = np.random.default_rng(seed)

    cruise = p.max_speed
    # per-trial placement jitter: train and eval draw different pad/gate
    # positions, so evaluation holds out WORLDS, not just trajectories, and
    # absolute distances cannot be memorised
    jitter = float(g.pad_jitter_m)
    pad_distance = float(g.pad_distance) + rng.uniform(-jitter, jitter, size=n)
    gap_m = target_gap_steps * cruise * p.dt
    gate_distance = pad_distance + gap_m

    # phase boundaries (per episode)
    approach_m = float(g.approach_m)

    # EXACTLY 50/50 pad side, then shuffled: LEFT(-1) -> OPEN, RIGHT(+1) ->
    # CLOSED. Exact balance (not a coin flip) is what makes the chance floor
    # exactly 0.5 rather than 0.5 +- sampling noise, so a majority-class
    # predictor cannot beat chance even slightly.
    pad_side = np.where(np.arange(n) < n // 2, -1.0, 1.0)
    rng.shuffle(pad_side)
    pad_target = pad_side * float(g.pad_offset)

    reach = p.half_length + p.gate_half_length
    total_m = float(gate_distance[0]) + float(g.margin_m)
    T = int(total_m / (0.90 * cruise * p.dt)) + 120

    states = np.zeros((n, T + 1, STATE_DIM), dtype=np.float32)
    actions = np.zeros((n, T), dtype=np.int8)

    s = np.zeros((n, STATE_DIM))
    s[:, SPEED] = p.initial_speed
    states[:, 0] = s

    steer_noise = float(g.steer_noise)   # rad/s of heading process noise
    noise_p = rng.uniform(0.0, 0.06, size=n)
    wander_target = rng.uniform(-2.0, 2.0, size=n)
    wander_period = int(g.wander_period)

    for t in range(T):
        d, x, h = s[:, DISTANCE], s[:, X], s[:, HEADING]

        if t % wander_period == 0:
            fresh = rng.uniform(-2.0, 2.0, size=n)
            wander_target = np.where(rng.random(n) < 0.7, fresh, wander_target)

        in_pad_approach = (d >= pad_distance - approach_m) & (d < pad_distance)
        past_pad = d >= pad_distance
        target = np.where(in_pad_approach, pad_target, np.where(past_pad, 0.0, wander_target))

        a = _steer_toward(x, h, target)
        # action noise everywhere except the two phases that must be reliable
        noisy = (rng.random(n) < noise_p) & ~in_pad_approach & ~past_pad
        a = np.where(noisy, rng.integers(0, 3, size=n), a)

        actions[:, t] = a
        noise = rng.normal(0.0, steer_noise, size=n) * p.dt
        s = step_batch(s, a.astype(int), p, pad_distance, gate_distance, heading_noise=noise)
        states[:, t + 1] = s

    dist = states[:, :, DISTANCE].astype(np.float64)
    # t_hit: first step at which the car is within the gate's reach. The model
    # must predict state[t_hit] from inputs up to t_hit-1, where the gate has
    # not yet acted on the car.
    at_gate = np.abs(dist - gate_distance[:, None]) < reach
    t_hit = np.argmax(at_gate, axis=1)
    reached = at_gate.any(axis=1)
    past_pad = dist >= pad_distance[:, None]
    t_pad = np.argmax(past_pad, axis=1)

    # trim the tail: nothing past the last gate crossing is used by any probe
    T_used = int(t_hit.max()) + 15
    states, actions = states[:, : T_used + 1], actions[:, :T_used]

    return {
        "states": states,
        "actions": actions,
        "pad_distance": pad_distance.astype(np.float32),
        "gate_distance": gate_distance.astype(np.float32),
        "t_hit": t_hit.astype(np.int32),
        "t_pad": t_pad.astype(np.int32),
        "reached": reached,
        "gate_open": (states[np.arange(n), t_hit, BIT] > 0.5),
        "target_gap_steps": target_gap_steps,
        "realised_gap": (t_hit - t_pad).astype(np.int32),
    }
