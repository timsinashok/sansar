"""The engine interface.

Every world simulator — the hand-written classic engine and, later, the
neural one — implements this. The game loop, data collection, and evals are
written against it, so swapping physics for a transformer is a one-line
change.

`step` is a pure function of (state, action): no hidden internal state. That
keeps trajectory generation trivially parallel/replayable, and it is the
contract the neural engine will be trained to match. Engines that need
history or a latent (the neural engine eventually) manage it behind this same
interface via `reset`.
"""

from typing import Protocol

from sansar.core.types import Action, CarState


class Engine(Protocol):
    def reset(self) -> CarState:
        """Start a new episode and return the initial state."""
        ...

    def step(self, state: CarState, action: Action) -> CarState:
        """Advance the world by one fixed timestep."""
        ...
