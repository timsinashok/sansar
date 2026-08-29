"""The dynamics transformer.

Context-as-memory (architecture A from the plan): a window of K past
(state-features + action) tokens goes through a transformer; the last
position's output predicts the next transition as 4 normalized state deltas
plus a collision logit. Only the last position is read, so attention over the
full window is causal for that prediction by construction.
"""

import torch
from omegaconf import DictConfig
from torch import nn

from sansar.models.features import INPUT_DIM

OUTPUT_DIM = 5  # 4 normalized deltas (x, heading, speed, distance) + collision logit


class DynamicsTransformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        d = int(cfg.d_model)
        self.context = int(cfg.context)
        self.proj = nn.Linear(INPUT_DIM, d)
        self.pos = nn.Parameter(torch.zeros(1, self.context, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.n_heads),
            dim_feedforward=4 * d,
            dropout=float(cfg.dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(cfg.n_layers), enable_nested_tensor=False
        )
        self.head = nn.Linear(d, OUTPUT_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, K, INPUT_DIM) -> (B, OUTPUT_DIM) for the next step."""
        h = self.proj(tokens) + self.pos[:, : tokens.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1])
