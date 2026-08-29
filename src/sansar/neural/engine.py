"""The neural engine: a trained dynamics transformer behind the same Engine
interface as the classic physics — the drop-in swap the whole project is
about.

It keeps an internal context window of the trajectory it has predicted so
far (the game loop's `state` argument is trusted to be its own last output;
`reset` clears history). Geometry and normalization stats come from the
checkpoint, so the engine simulates the world it was trained on regardless
of the current config.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from sansar.core.types import Action, CarState
from sansar.envs.driving.physics import ClassicEngine
from sansar.models.dynamics import DynamicsTransformer
from sansar.models.features import Geometry, integrate, tokens


class NeuralEngine:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model_cfg = OmegaConf.create(ckpt["model_cfg"])
        self.env_cfg = OmegaConf.create(ckpt["env_cfg"])
        self.device = device
        self.context = int(model_cfg.context)

        self.model = DynamicsTransformer(model_cfg).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.stats = {k: v.to(device) for k, v in ckpt["stats"].items()}
        self.geom = Geometry(self.env_cfg, device)

        # classic engine used ONLY for the initial state at reset
        self._classic = ClassicEngine(self.env_cfg)
        self._states: torch.Tensor | None = None   # (K, 5)
        self._actions: torch.Tensor | None = None  # (K,)

    def reset(self) -> CarState:
        s0 = self._classic.reset()
        arr = torch.from_numpy(s0.to_array()).to(self.device)
        self._states = arr.unsqueeze(0).repeat(self.context, 1)
        self._actions = torch.zeros(self.context, dtype=torch.long, device=self.device)
        return s0

    @torch.no_grad()
    def step(self, state: CarState, action: Action) -> CarState:
        if self._states is None:
            self.reset()
        # the last window slot is the current state; stamp it with the action
        # actually taken, matching training alignment (token t = s_t, a_t)
        self._actions[-1] = int(action)
        toks = tokens(
            self.geom, self.stats, self._states.unsqueeze(0), self._actions.unsqueeze(0)
        )
        out = self.model(toks)
        nxt = integrate(self._states[-1].unsqueeze(0), out, self.stats)[0]
        self._states = torch.cat([self._states[1:], nxt.unsqueeze(0)])
        self._actions = torch.cat(
            [self._actions[1:], torch.zeros(1, dtype=torch.long, device=self.device)]
        )
        return CarState.from_array(nxt.cpu().numpy())
