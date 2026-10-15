"""Eager independent-chain samplers. Cached targets must remain deterministic.

Call invalidate_cache() after changing target data/parameters. Results and public
position/momentum properties are owned snapshots. State dicts include configuration
and generator state; exact continuation requires the same target, device and stack.
"""
import copy
import math
import torch
from ..targets import value_and_grad, validate_position
from ..state import StepResult
from ..functional import ula_step, mala_log_acceptance, lip_step_size, ou_step


def _tensor(value, x, name, *, positive=True):
    out = torch.as_tensor(value, device=x.device, dtype=x.dtype).detach().clone()
    invalid_sign = (out <= 0).any() if positive else (out < 0).any()
    if not torch.isfinite(out).all() or invalid_sign:
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return out


class ULA:
    corrected = False

    def __init__(self, potential, initial, step_size=.01, *, temperature=1., mobility=1., generator=None):
        validate_position(initial)
        self.potential_fn = potential
        self._position = initial.detach().clone()
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        self.mobility = _tensor(mobility, initial, 'mobility')
        if torch.broadcast_shapes(self.mobility.shape, initial.shape) != initial.shape:
            raise ValueError('mobility must broadcast to position')
        h = _tensor(step_size, initial, 'step_size')
        try:
            self.step_size = h.expand(initial.shape[:-1]).clone()
        except RuntimeError as exc:
            raise ValueError('step_size must be scalar or one value per chain') from exc
        self.generator = generator if generator is not None else torch.Generator(device=initial.device)
        if torch.device(self.generator.device) != initial.device:
            raise ValueError('generator must be on the same device as initial')
        if generator is None:
            self.generator.seed()
        if self.corrected and getattr(potential, 'score_only', False):
            raise ValueError('MALA requires potential values, not only a score')
        self.iteration = 0
        self.invalidate_cache()

    @property
    def position(self):
        return self._position.clone()

    def _evaluate(self, x):
        return value_and_grad(self.potential_fn, x)

    def invalidate_cache(self):
        value, gradient = self._evaluate(self._position)
        if not torch.isfinite(gradient).all() or (not getattr(self.potential_fn, 'score_only', False) and not torch.isfinite(value).all()):
            raise ValueError('current potential and gradient must be finite')
        self._value, self._gradient = value, gradient

    def _noise(self):
        return torch.randn(self._position.shape, device=self._position.device, dtype=self._position.dtype, generator=self.generator)

    def _h(self):
        return self.step_size.unsqueeze(-1)

    def _snapshot(self, accepted=None, **stats):
        return StepResult(self.position, self._value.clone(), self._gradient.clone(),
                          None if accepted is None else accepted.clone(),
                          {k: torch.as_tensor(v).detach().clone() for k,v in stats.items()})

    def step(self):
        x, g = self._position, self._gradient
        proposal = ula_step(x, g, self._h(), temperature=self.temperature, mobility=self.mobility, noise=self._noise())
        if self.corrected:
            coordinate_finite = torch.isfinite(proposal).all(-1)
            # Invalid chains are evaluated at their valid current state, then
            # forced to reject. This also protects optimizer closures before
            # any invalid full-state proposal is assigned to parameters.
            evaluation_position = torch.where(coordinate_finite.unsqueeze(-1), proposal, x)
        else:
            evaluation_position = proposal
        v_new, g_new = self._evaluate(evaluation_position)
        accepted = None
        stats = {'step_size': self.step_size}
        if self.corrected:
            ratio = mala_log_acceptance(x, proposal, self._value, v_new, g, g_new, self._h(),
                                        temperature=self.temperature, mobility=self.mobility)
            finite = coordinate_finite & torch.isfinite(v_new) & torch.isfinite(g_new).all(-1) & torch.isfinite(ratio)
            uniform = torch.rand(ratio.shape, device=x.device, dtype=x.dtype, generator=self.generator)
            accepted = finite & (uniform.log() < ratio.clamp_max(0))
            self._position = torch.where(accepted.unsqueeze(-1), proposal, x)
            self._gradient = torch.where(accepted.unsqueeze(-1), g_new, g)
            self._value = torch.where(accepted, v_new, self._value)
            stats['log_acceptance'] = torch.where(finite, ratio.clamp_max(0), torch.full_like(ratio, -float('inf')))
        else:
            if not torch.isfinite(proposal).all() or not torch.isfinite(g_new).all() or (not getattr(self.potential_fn,'score_only',False) and not torch.isfinite(v_new).all()):
                raise ValueError('nonfinite ULA transition; reduce step size or inspect target')
            self._position, self._value, self._gradient = proposal.detach(), v_new, g_new
        self.iteration += 1
        return self._snapshot(accepted, **stats)

    def state_dict(self):
        data = {k: copy.deepcopy(v) for k,v in self.__dict__.items() if k not in ('potential_fn','generator')}
        return {'schema_version': 1, 'algorithm': type(self).__name__, 'state': data,
                'generator_state': self.generator.get_state().clone(), 'device': str(self._position.device)}

    def load_state_dict(self, checkpoint):
        if checkpoint['schema_version'] != 1 or checkpoint['algorithm'] != type(self).__name__:
            raise ValueError('incompatible checkpoint schema or algorithm')
        state = copy.deepcopy(checkpoint['state'])
        x = state['_position']
        if x.shape != self._position.shape or x.dtype != self._position.dtype or checkpoint['device'] != str(self._position.device):
            raise ValueError('checkpoint requires matching shape, dtype and device')
        validate_position(x)
        self.__dict__.update(state)
        self.generator.set_state(checkpoint['generator_state'])


class MALA(ULA):
    """Fixed-covariance MALA with independent whole-chain MH decisions."""
    corrected = True


class _Lip:
    """History adaptation is approximate; adapt_steps freezes production h.

    max_step_size (finite default 1) safeguards constant-gradient targets. A
    rejection/repeated state leaves h and its ratio history unchanged. alpha starts
    at infinity (the paper initialization), then equals the latest h/previous h. Each chain adapts separately.
    """
    def __init__(self, *args, lc=None, max_step_size=1., adapt_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        if adapt_steps is not None and (not isinstance(adapt_steps,int) or adapt_steps < 0):
            raise ValueError('adapt_steps must be a nonnegative integer or None')
        self.adapt_steps = adapt_steps
        self.lc = self._position.shape[-1]**(-1/3) if lc is None else float(lc)
        self.max_step_size = float(max_step_size)
        if not math.isfinite(self.lc) or self.lc <= 0 or not math.isfinite(self.max_step_size) or self.max_step_size <= 0:
            raise ValueError('lc and max_step_size must be positive finite numbers')
        if (self.step_size > self.max_step_size).any():
            raise ValueError('initial step_size exceeds max_step_size')
        self._alpha = torch.full_like(self.step_size, float('inf'))

    def step(self):
        old_x, old_g, old_h = self._position.clone(), self._gradient.clone(), self.step_size.clone()
        result = super().step()
        if self.adapt_steps is None or self.iteration <= self.adapt_steps:
            self.step_size = lip_step_size(self._position, old_x, self._gradient, old_g, old_h, self._alpha,
                                          mobility=self.mobility, lc=self.lc, max_step_size=self.max_step_size)
            moved = (self._position != old_x).any(-1)
            self._alpha = torch.where(moved, self.step_size/old_h, self._alpha)
        result.stats['next_step_size'] = self.step_size.clone()
        result.stats['adapting'] = torch.tensor(self.adapt_steps is None or self.iteration < self.adapt_steps, device=self._position.device)
        return result


class LipULA(_Lip, ULA):
    """ULA with safeguarded local Lipschitz adaptation."""


class LipMALA(_Lip, MALA):
    """MALA moves with history adaptation; freeze h for fixed-kernel production."""


class BAOAB(ULA):
    """B-A-O-A-B splitting, with dp=-grad(U)dt-damping*p dt+sqrt(2*damping*mass*T)dW."""
    def __init__(self, *args, damping=1., mass=1., momentum=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not torch.equal(self.mobility, torch.ones_like(self.mobility)):
            raise ValueError('BAOAB uses mass, not mobility')
        self.damping = _tensor(damping, self._position, 'damping', positive=False)
        self.mass = _tensor(mass, self._position, 'mass')
        for name in ('damping','mass'):
            if torch.broadcast_shapes(getattr(self,name).shape,self._position.shape) != self._position.shape:
                raise ValueError(f'{name} must broadcast to position')
        self._momentum = torch.zeros_like(self._position) if momentum is None else momentum.detach().clone()
        if self._momentum.shape != self._position.shape or self._momentum.dtype != self._position.dtype or self._momentum.device != self._position.device or not torch.isfinite(self._momentum).all():
            raise ValueError('momentum must match finite position shape, dtype and device')

    @property
    def momentum(self):
        return self._momentum.clone()

    def step(self):
        h = self._h()
        p = self._momentum - h*self._gradient/2
        x = self._position + h*p/(2*self.mass)
        p = ou_step(p, h, damping=self.damping, mass=self.mass, temperature=self.temperature, noise=self._noise())
        x = x + h*p/(2*self.mass)
        v,g = self._evaluate(x)
        p = p-h*g/2
        self._finish_kinetic(x,v,g,p)
        return self._snapshot(step_size=self.step_size, momentum=self._momentum)

    def _finish_kinetic(self,x,v,g,p):
        if not torch.isfinite(x).all() or not torch.isfinite(p).all() or not torch.isfinite(g).all() or (not getattr(self.potential_fn,'score_only',False) and not torch.isfinite(v).all()):
            raise ValueError('nonfinite kinetic transition')
        self._position,self._value,self._gradient,self._momentum = x.detach(),v,g,p.detach()
        self.iteration += 1


class KineticLangevin(BAOAB):
    """Explicit Euler-Maruyama underdamped integrator (distinct from BAOAB)."""
    def step(self):
        h=self._h()
        x=self._position+h*self._momentum/self.mass
        p=self._momentum-h*self._gradient-h*self.damping*self._momentum+(2*h*self.damping*self.mass*self.temperature).sqrt()*self._noise()
        v,g=self._evaluate(x)
        self._finish_kinetic(x,v,g,p)
        return self._snapshot(step_size=self.step_size,momentum=self._momentum)
