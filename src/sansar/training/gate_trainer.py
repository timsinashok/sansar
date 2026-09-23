"""Training for the gate memory experiment.

Teacher-forced throughout: the model always sees ground-truth observation
history. This is deliberate — Probe 1 measures MEMORY, and feeding the model
its own predictions would confound retention failure with rollout drift.
Probe 2 (eval/stability.py) measures drift separately, on free rollout.

Window and recurrent models take different fast paths to the same loss:

    window     sample (trial, idx) pairs; half the batch is anchored on
               idx = t_hit so the readout is actually trained (gate steps are
               ~0.1% of a trial otherwise)
    recurrent  one fused nn.GRU pass over whole trials; loss at every step,
               with the gate step upweighted

Deltas use MSE rather than Huber: Huber's gradient saturates past 1 sigma, so
on a normalised delta target it systematically under-weights exactly the
large corrections that matter (see docs/V05_PLAN.md section 1).
"""

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from sansar.envs.gate.vec import collect_trials
from sansar.envs.gate.world import COLLIDED
from sansar.models.gate_features import compute_stats, features, tokens
from sansar.models.sequence import build


class TrialSet:
    """Tokenised trials plus the gate readout index for each."""

    def __init__(self, raw: dict, stats: dict | None, device: str):
        st = torch.from_numpy(raw["states"]).float()
        ac = torch.from_numpy(raw["actions"]).long()
        pad = torch.from_numpy(raw["pad_distance"]).float()
        gate = torch.from_numpy(raw["gate_distance"]).float()
        feats = features(st[:, :-1], pad, gate)
        self.stats = compute_stats(feats, st) if stats is None else stats
        self.tokens = tokens(feats, ac, self.stats).to(device)
        self.states = st.to(device)
        self.actions = ac.to(device)
        self.pad_d = pad.to(device)
        self.gate_d = gate.to(device)
        self.t_hit = torch.from_numpy(raw["t_hit"]).long().to(device)
        self.gate_open = torch.from_numpy(raw["gate_open"]).bool().to(device)
        self.label = self.states[torch.arange(len(self.t_hit)), self.t_hit, COLLIDED]
        self.device = device

    def __len__(self):
        return self.tokens.shape[0]


def _targets(ts: TrialSet, trial: torch.Tensor, idx: torch.Tensor):
    cur = ts.states[trial, idx - 1]
    nxt = ts.states[trial, idx]
    delta = (nxt[:, :4] - cur[:, :4] - ts.stats["delta_mean"].to(ts.device)) / ts.stats[
        "delta_std"
    ].to(ts.device)
    return delta, nxt[:, COLLIDED]


def _loss(out, delta_t, coll_t, w):
    return F.mse_loss(out[:, :4], delta_t) + w * F.binary_cross_entropy_with_logits(
        out[:, 4], coll_t
    )


@torch.no_grad()
def gate_accuracy(model, ts: TrialSet, batch: int = 256) -> float:
    """Probe 1: does the collision head fire correctly at the gate?"""
    model.eval()
    correct = 0
    for i in range(0, len(ts), batch):
        sl = slice(i, min(i + batch, len(ts)))
        out = model.predict_at(ts.tokens[sl], ts.t_hit[sl])
        correct += int(((out[:, 4] > 0).float() == ts.label[sl]).sum())
    model.train()
    return correct / len(ts)


def train_one(
    name: str, gap: int, seed: int, cfg: DictConfig, device: str,
    verbose: bool = False, return_model: bool = False,
):
    """Train one (architecture, gap, seed) cell and return its metrics."""
    torch.manual_seed(seed)
    sw, tcfg = cfg.sweep, cfg.gate_train

    # train and eval draw disjoint placement jitter and driver randomness
    tr_raw = collect_trials(cfg.env, int(sw.train_trials), gap, seed=1000 * seed + gap)
    ev_raw = collect_trials(cfg.env, int(sw.eval_trials), gap, seed=500_000 + 1000 * seed + gap)
    tr = TrialSet(tr_raw, None, device)
    ev = TrialSet(ev_raw, tr.stats, device)

    model = build(name, cfg.gate_model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg.lr), weight_decay=float(tcfg.weight_decay)
    )
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, T = tr.tokens.shape[0], tr.tokens.shape[1]
    B = int(tcfg.batch_size)
    w = float(tcfg.collision_loss_weight)
    best, best_state = 0.0, None

    for step in range(1, int(tcfg.steps) + 1):
        if model.kind == "window":
            trial = torch.randint(0, N, (B,), generator=g).to(device)
            rand_idx = torch.randint(1, T, (B,), generator=g).to(device)
            on_gate = torch.rand(B, generator=g).to(device) < float(tcfg.gate_sample_frac)
            idx = torch.where(on_gate, tr.t_hit[trial], rand_idx)
            out = model.predict_at(tr.tokens[trial], idx)
            delta_t, coll_t = _targets(tr, trial, idx)
            loss = _loss(out, delta_t, coll_t, w)
        else:
            trial = torch.randint(0, N, (int(tcfg.gru_batch),), generator=g).to(device)
            tok = tr.tokens[trial]
            outs = model.forward_sequence(tok)                    # (b, T, 5): out[t] -> state t+1
            cur, nxt = tr.states[trial, :-1], tr.states[trial, 1:]
            dm, ds = tr.stats["delta_mean"].to(device), tr.stats["delta_std"].to(device)
            delta_t = (nxt[..., :4] - cur[..., :4] - dm) / ds
            coll_t = nxt[..., COLLIDED]
            cw = torch.ones_like(coll_t)
            cw[torch.arange(len(trial), device=device), tr.t_hit[trial] - 1] = float(
                tcfg.gate_step_weight
            )
            loss = F.mse_loss(outs[..., :4], delta_t) + w * (
                F.binary_cross_entropy_with_logits(outs[..., 4], coll_t, reduction="none") * cw
            ).sum() / cw.sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % int(tcfg.eval_every) == 0 or step == int(tcfg.steps):
            acc = gate_accuracy(model, ev)
            if acc > best:
                best = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"    [{name} G={gap} s={seed}] {step:5d} loss {loss.item():.4f} gate {acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "model": name,
        "gap": gap,
        "seed": seed,
        "gate_acc": best,
        "final_acc": gate_accuracy(model, ev),
        "n_params": model.n_params(),
        "realised_gap": int(np.median(ev_raw["realised_gap"])),
        "n_eval": len(ev),
    }
    return (metrics, model, ev) if return_model else metrics
