"""Potential convention U=-log(pi). Batched targets must separate chains."""
from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor

Potential = Callable[[Tensor], Tensor]


def validate_position(x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.layout != torch.strided or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('position must be a dense float32 or float64 tensor')
    if x.ndim not in (1, 2) or 0 in x.shape or not torch.isfinite(x).all():
        raise ValueError('position must be finite with shape (D,) or (chains, D)')


def value_and_grad(potential: Potential, x: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate without changing x/.grad or retaining graphs.

    Batch independence is a caller contract: values must have shape x.shape[:-1]
    and each value must depend only on its corresponding chain.
    """
    if hasattr(potential, 'value_and_grad'):
        value, gradient = potential.value_and_grad(x.detach().clone())
    else:
        with torch.enable_grad():
            leaf = x.detach().clone().requires_grad_(True)
            value = potential(leaf)
            if not isinstance(value, Tensor) or value.shape != x.shape[:-1]:
                raise ValueError('potential must return one scalar per chain, shape x.shape[:-1]')
            gradient = torch.autograd.grad(value.sum(), leaf, allow_unused=True)[0] if value.requires_grad else None
            if gradient is None:
                gradient = torch.zeros_like(leaf)
    if value.shape != x.shape[:-1] or gradient.shape != x.shape:
        raise ValueError('target value/gradient shapes do not match position')
    if value.device != x.device or gradient.device != x.device or value.dtype != x.dtype or gradient.dtype != x.dtype:
        raise ValueError('target must preserve position dtype and device')
    return value.detach().clone(), gradient.detach().clone()


def from_log_prob(log_prob: Potential) -> Potential:
    """Convert log probability to potential without changing temperature."""
    return lambda x: -log_prob(x)


@dataclass
class ScoreTarget:
    score: Callable[[Tensor], Tensor]
    score_only: bool = True

    def value_and_grad(self, x):
        return x.new_full(x.shape[:-1], float('nan')), -self.score(x)


def from_score(score: Callable[[Tensor], Tensor]) -> ScoreTarget:
    """Gradient-only target. Potential snapshots are NaN; cannot be used by MALA."""
    return ScoreTarget(score)


def positive_transform(potential: Potential) -> Potential:
    """Pull back a density on positive coordinates via z=exp(x), including Jacobian."""
    return lambda x: potential(x.exp()) - x.sum(-1)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: torch.Size
    start: int
    stop: int


class ModulePotential:
    """Functional module evaluation for one flat parameter vector.

    loss(output) returns scalar U. Buffers are snapshotted at construction and
    cloned for each call, so model buffer mutation cannot leak between calls.
    The caller must use deterministic model evaluation (e.g. model.eval()).
    Module parameters are never overwritten. Batched vectors are not supported.
    """
    def __init__(self, module, loss, *, args=(), kwargs=None):
        self.module, self.loss, self.args = module, loss, args
        self.kwargs = {} if kwargs is None else dict(kwargs)
        self.parameter_spec = []
        offset = 0
        for name, p in module.named_parameters():
            self.parameter_spec.append(ParameterSpec(name, p.shape, offset, offset+p.numel()))
            offset += p.numel()
        if not offset:
            raise ValueError('module must have parameters')
        self.numel = offset
        self.buffers = {n: b.detach().clone() for n, b in module.named_buffers()}

    def flatten_parameters(self):
        return torch.cat([p.detach().reshape(-1) for p in self.module.parameters()]).clone()

    def __call__(self, x):
        if x.ndim != 1 or x.numel() != self.numel:
            raise ValueError('ModulePotential requires one flat vector of the specified size')
        params = {s.name: x[s.start:s.stop].view(s.shape) for s in self.parameter_spec}
        buffers = {n: b.to(device=x.device).clone() for n,b in self.buffers.items()}
        output = torch.func.functional_call(self.module, (params, buffers), self.args, self.kwargs)
        return self.loss(output)
