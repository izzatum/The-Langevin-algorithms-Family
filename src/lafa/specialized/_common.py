"""Shared validation and owned state for specialized algorithms."""
import copy
import math
from dataclasses import dataclass, field
import torch
from ..targets import validate_position


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def checked(value, like, name, shape=None):
    shape = like.shape if shape is None else shape
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != like.dtype or value.device != like.device or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite and preserve shape, dtype and device')
    return value.detach().clone()


def noise_like(x, generator, supplied=None):
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) if supplied is None else checked(supplied, x, 'noise')


@dataclass
class SpecializedResult:
    position: torch.Tensor
    stats: dict = field(default_factory=dict)


@dataclass
class InteractingResult:
    theta: torch.Tensor
    particles: torch.Tensor
    stats: dict = field(default_factory=dict)


class OwnedState:
    _state_fields = ('_position', 'iteration', 'step_size')

    def _initialize(self, x, generator):
        validate_position(x)
        self._position = x.detach().clone()
        self.iteration = 0
        self.generator = generator if generator is not None else torch.Generator(device=x.device)
        if torch.device(self.generator.device) != x.device:
            raise ValueError('generator must match state device')
        if generator is None:
            self.generator.seed()

    @property
    def position(self):
        return self._position.detach().clone()

    def state_dict(self):
        return {'schema_version': 1, 'algorithm': type(self).__name__,
                'state': {k: copy.deepcopy(getattr(self, k)) for k in self._state_fields},
                'generator_state': self.generator.get_state().clone()}

    def load_state_dict(self, checkpoint):
        if checkpoint['schema_version'] != 1 or checkpoint['algorithm'] != type(self).__name__:
            raise ValueError('incompatible checkpoint')
        state = copy.deepcopy(checkpoint['state'])
        checked(state['_position'], self._position, 'checkpoint position')
        if set(state) != set(self._state_fields):
            raise ValueError('incompatible checkpoint fields')
        self.generator.set_state(checkpoint['generator_state'])
        self.__dict__.update(state)

    def _finish(self, x, **stats):
        self._position = checked(x, self._position, 'transition')
        self.iteration += 1
        return SpecializedResult(self.position, {k: v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k,v in stats.items()})
