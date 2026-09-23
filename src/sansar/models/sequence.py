"""The control ladder: four architectures behind one interface.

Every model answers the same question — "given the observation-action history
up to step i-1, what is state i?" — so the memory and stability probes are
written once and applied to all of them.

    memoryless   window of 1     the leak control; must score chance
    mlp          flat window K   does attention matter, or just history?
    transformer  window K        the V0 architecture
    gru          carried latent  can recurrence beat a window at equal params?

`predict_at(tokens, idx)` is the shared entry point. Window models slice
tokens[idx-K:idx]; the GRU runs its recurrence over tokens[:idx]. Both return
(B, OUTPUT_DIM).
"""

import torch
from omegaconf import DictConfig
from torch import nn

from sansar.models.gate_features import INPUT_DIM, OUTPUT_DIM


class _Base(nn.Module):
    kind = "window"

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_window(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # --- incremental interface, used by the free-rollout stability probe ---

    def init_mem(self, batch: int, device: str):
        return torch.zeros(batch, self.context, INPUT_DIM, device=device)

    def step_token(self, token: torch.Tensor, mem):
        mem = torch.cat([mem[:, 1:], token[:, None]], dim=1)
        return self.forward_window(mem), mem

    def predict_at(self, tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """tokens (B, T, D), idx (B,) -> (B, OUTPUT_DIM) prediction of state idx.

        Uses tokens strictly before idx. Window models left-pad by repeating
        the earliest available token when idx < K.
        """
        B, T, D = tokens.shape
        K = self.context
        ar = torch.arange(K, device=tokens.device)
        # positions idx-K .. idx-1, clamped into range
        pos = (idx[:, None] - K + ar[None, :]).clamp(min=0, max=T - 1)
        w = torch.gather(tokens, 1, pos[..., None].expand(-1, -1, D))
        return self.forward_window(w)


class MemorylessMLP(_Base):
    """Window of 1. Provably at chance on the gate task if the task is clean."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.context = 1
        h = int(cfg.mlp_hidden)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, OUTPUT_DIM),
        )

    def forward_window(self, w):
        return self.net(w[:, -1])


class WindowMLP(_Base):
    """Flattened K-step window. Isolates attention from mere history access."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.context = int(cfg.context)
        h = int(cfg.mlp_hidden)
        self.net = nn.Sequential(
            nn.Linear(self.context * INPUT_DIM, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, OUTPUT_DIM),
        )

    def forward_window(self, w):
        return self.net(w.flatten(1))


class WindowTransformer(_Base):
    """The V0 architecture, unchanged in form: attention over a K-step window."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.context = int(cfg.context)
        d = int(cfg.d_model)
        self.proj = nn.Linear(INPUT_DIM, d)
        self.pos = nn.Parameter(torch.zeros(1, self.context, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=int(cfg.n_heads), dim_feedforward=4 * d,
            dropout=float(cfg.dropout), batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(cfg.n_layers), enable_nested_tensor=False
        )
        self.head = nn.Linear(d, OUTPUT_DIM)

    def forward_window(self, w):
        h = self.proj(w) + self.pos[:, : w.shape[1]]
        return self.head(self.encoder(h)[:, -1])


class RecurrentLatent(_Base):
    """A single carried latent z_t — the thesis formulation.

    No window: information reaches the gate only if the recurrence retained
    it. Sequence-major so nn.GRU's fused kernel does the time loop.
    """

    kind = "recurrent"

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.context = 1
        d = int(cfg.gru_hidden)
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.gru = nn.GRU(d, d, num_layers=int(cfg.gru_layers), batch_first=True)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, OUTPUT_DIM))

    def forward_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, OUTPUT_DIM); output t predicts state t+1."""
        h, _ = self.gru(self.encoder(tokens))
        return self.head(h)

    def predict_at(self, tokens, idx):
        outs = self.forward_sequence(tokens)
        return outs[torch.arange(tokens.shape[0], device=tokens.device), (idx - 1).clamp(min=0)]

    def init_mem(self, batch: int, device: str):
        return torch.zeros(
            self.gru.num_layers, batch, self.gru.hidden_size, device=device
        )

    def step_token(self, token: torch.Tensor, mem):
        h, mem = self.gru(self.encoder(token)[:, None], mem)
        return self.head(h[:, 0]), mem


BUILDERS = {
    "memoryless": MemorylessMLP,
    "mlp": WindowMLP,
    "transformer": WindowTransformer,
    "gru": RecurrentLatent,
}


def build(name: str, cfg: DictConfig) -> _Base:
    return BUILDERS[name](cfg)
