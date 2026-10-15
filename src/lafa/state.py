"""Detached, independently owned transition snapshots; no trajectory retention."""
from dataclasses import dataclass, field
import torch

@dataclass
class StepResult:
    """End-of-step state. Sampler-produced tensors are copies safe to retain/mutate."""
    position: torch.Tensor
    potential: torch.Tensor
    gradient: torch.Tensor
    accepted: torch.Tensor | None = None
    stats: dict[str, torch.Tensor] = field(default_factory=dict)
