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


def _validate_chunk_size(chunk_size):
    if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError('chunk_size must be None or a positive integer')


def _transform_error(error, remedy):
    """Annotate recognized transform limitations without hiding model errors."""
    message = str(error)
    if any(term in message.lower() for term in ('vmap', 'batchedtensor', 'functorch', 'data-dependent', '.item()', 'randomness', 'being transformed')):
        raise RuntimeError(f'Unsupported transformed operation; use {remedy}. Original error: {message}') from error
    raise error


class ModulePotential:
    """Independent functional evaluation of flat (D,) or (chains,D) states.

    loss(output) returns scalar U. Buffers are snapshotted at construction and
    cloned for each call, so model buffer mutation cannot leak between calls.
    The caller must use deterministic model evaluation (e.g. model.eval()).
    Module parameters are never overwritten. ``batch_mode='loop'`` supports
    operations that cannot be transformed by vmap; no automatic fallback occurs.
    ``chunk_size`` bounds chain activations during ``value_and_grad``. Ordinary
    differentiable calls retain their graphs until the caller releases them.
    Same-object tied parameters retain their tie. Distinct parameters sharing
    storage are rejected because an unconstrained flat vector loses that alias.
    """
    def __init__(self, module, loss, *, args=(), kwargs=None, batch_mode='vmap', chunk_size=None):
        if batch_mode not in ('vmap', 'loop'):
            raise ValueError("batch_mode must be 'vmap' or 'loop'")
        _validate_chunk_size(chunk_size)
        self.batch_mode, self.chunk_size = batch_mode, chunk_size
        self.module, self.loss, self.args = module, loss, args
        self.kwargs = {} if kwargs is None else dict(kwargs)
        self.parameter_spec = []
        offset = 0
        seen_storage = set()
        # named_parameters removes duplicate objects, preserving ordinary tied
        # weights. Distinct objects backed by one storage need an unsupported
        # constraint and cannot become independent flat coordinates.
        for name, p in module.named_parameters():
            storage = (p.device, p.untyped_storage().data_ptr())
            if storage in seen_storage:
                raise ValueError('distinct module parameters with aliased parameter storage are unsupported')
            seen_storage.add(storage)
            self.parameter_spec.append(ParameterSpec(name, p.shape, offset, offset+p.numel()))
            offset += p.numel()
        if not offset:
            raise ValueError('module must have parameters')
        self.numel = offset
        self.buffers = {n: b.detach().clone() for n, b in module.named_buffers()}

    def flatten_parameters(self):
        return torch.cat([p.detach().reshape(-1) for p in self.module.parameters()]).clone()

    def _validate(self, x):
        validate_position(x)
        if x.shape[-1] != self.numel:
            raise ValueError('ModulePotential requires the specified flat parameter size')

    def _single(self, x, buffers):
        # Parameter mutation inside a module must never overwrite caller states.
        owned = x.clone()
        params = {s.name: owned[s.start:s.stop].view(s.shape) for s in self.parameter_spec}
        output = torch.func.functional_call(self.module, (params, buffers), self.args, self.kwargs)
        value = self.loss(output)
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise ValueError('ModulePotential loss must return one scalar per state')
        if value.dtype != x.dtype or value.device != x.device:
            raise ValueError('ModulePotential loss must preserve position dtype and device')
        return value.clone()

    def _buffers(self, x, count=None):
        return {n: (b.to(device=x.device).clone() if count is None else
                    b.to(device=x.device).expand((count,)+b.shape).clone())
                for n, b in self.buffers.items()}

    def _batch(self, x):
        if self.batch_mode == 'loop':
            return torch.stack([self._single(row, self._buffers(row)) for row in x])
        try:
            return torch.vmap(self._single)(x, self._buffers(x, len(x)))
        except RuntimeError as error:
            _transform_error(error, "batch_mode='loop'")

    def __call__(self, x):
        self._validate(x)
        if x.ndim == 1:
            return self._single(x, self._buffers(x))
        size = self.chunk_size or len(x)
        return torch.cat([self._batch(x[i:i+size]) for i in range(0, len(x), size)])

    def value_and_grad(self, x):
        """Owned detached values and gradients, releasing each chunk's graph."""
        self._validate(x)
        rows = x.unsqueeze(0) if x.ndim == 1 else x
        values, gradients = [], []
        size = self.chunk_size or len(rows)
        for start in range(0, len(rows), size):
            with torch.enable_grad():
                leaf = rows[start:start+size].detach().clone().requires_grad_(True)
                value = (self._single(leaf[0], self._buffers(leaf)).unsqueeze(0)
                         if x.ndim == 1 else self._batch(leaf))
                gradient = torch.autograd.grad(value.sum(), leaf, allow_unused=True)[0] if value.requires_grad else None
                if gradient is None:
                    gradient = torch.zeros_like(leaf)
            values.append(value.detach().clone())
            gradients.append(gradient.detach().clone())
        value, gradient = torch.cat(values), torch.cat(gradients)
        return (value[0], gradient[0]) if x.ndim == 1 else (value, gradient)
