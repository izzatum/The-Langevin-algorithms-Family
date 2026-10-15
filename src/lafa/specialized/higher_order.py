"""Exact-reference anchored Euler and source-defined HOLA."""
import math
import torch
from ..targets import value_and_grad
from ._common import OwnedState, positive, checked, noise_like


class AnchoredLangevin(OwnedState):
    """R03 Eqs. (16),(33): exact-reference Euler, unit temperature.

    U and U0 return one value per chain. Only U0 needs a gradient. The matched
    exponential multiplies drift and diffusion covariance; additive constants
    affect speed. No clipping or finite-MC smoothing is used. operational_time
    tracks the random clock; equal operational-time samples target U0 instead.
    Arbitrary reference choices need not satisfy the paper's coercivity bounds.
    """
    _state_fields = OwnedState._state_fields + ('_operational_time',)

    def __init__(self, potential, reference_potential, initial, step_size=.01, *, generator=None):
        self._initialize(initial, generator)
        self.potential = potential
        self.reference_potential = reference_potential
        self.step_size = positive(step_size, 'step_size')
        self._operational_time = initial.new_zeros(initial.shape[:-1])
        self._reference(initial)

    @property
    def operational_time(self):
        return self._operational_time.clone()

    def _reference(self, x):
        with torch.no_grad():
            u = self.potential(x)
        checked(u, x, 'potential', x.shape[:-1])
        v,g = value_and_grad(self.reference_potential, x)
        checked(v, x, 'reference potential', x.shape[:-1])
        checked(g, x, 'reference gradient')
        return u-v,g

    def step(self, *, noise=None):
        log_a,g = self._reference(self._position)
        log_clock = log_a + math.log(self.step_size)
        limits = torch.finfo(self._position.dtype)
        if (log_clock > math.log(limits.max)).any() or (log_clock < math.log(limits.tiny)).any():
            raise ValueError('anchored clock overflow/underflow; change reference additive constant or step size')
        delta = log_clock.exp()
        x = self._position-delta.unsqueeze(-1)*g+(2*delta).sqrt().unsqueeze(-1)*noise_like(self._position,self.generator,noise)
        new_clock = self._operational_time+delta
        if not torch.isfinite(new_clock).all():
            raise ValueError('operational clock overflow')
        result = self._finish(x, clock_increment=delta, operational_time=new_clock)
        self._operational_time = new_clock.detach()
        return result


class HOLA(OwnedState):
    """R07 Eqs. (2)-(4), or untamed Eq. (8), unit temperature U (use U/T).

    Dense autograd is a small-dimension reference, O(d²) storage plus third
    derivatives, guarded by max_dense_dimension. Batched chains loop explicitly.
    A matrix-free provider exposes gradient(x), hvp(x,v), spectral_norm(x), and
    grad_laplacian(x), all exact; approximations are not silently substituted.
    Four distinct tamings follow the paper. tamed=False requires the stronger
    globally Lipschitz assumptions; neither scheme is MH corrected.
    """
    _state_fields = OwnedState._state_fields + ('tamed', 'max_dense_dimension')

    def __init__(self, potential, initial, step_size=.01, *, tamed=True, generator=None, derivatives=None, max_dense_dimension=64):
        self._initialize(initial, generator)
        self.potential = potential
        self.step_size = positive(step_size, 'step_size')
        if self.step_size >= 1:
            raise ValueError('HOLA requires 0 < step_size < 1')
        self.tamed = bool(tamed)
        self.max_dense_dimension = int(max_dense_dimension)
        self.derivatives = derivatives
        if derivatives is None and initial.shape[-1] > self.max_dense_dimension:
            raise ValueError('dense HOLA dimension guard; supply exact derivative provider')

    def _derivatives(self, x):
        if self.derivatives is not None:
            provider = self.derivatives
            g = checked(provider.gradient(x), x, 'gradient')
            lap = checked(provider.grad_laplacian(x), x, 'gradient of Laplacian')
            norm = torch.as_tensor(provider.spectral_norm(x),device=x.device,dtype=x.dtype)
            if norm.ndim or not torch.isfinite(norm) or norm < 0:
                raise ValueError('exact spectral norm must be a finite nonnegative scalar')
            return g, lambda v: checked(provider.hvp(x,v), x, 'Hessian-vector product'), norm, lap
        with torch.enable_grad():
            # torch.func handles constant derivatives (linear/quadratic U).
            gradient = torch.func.grad(self.potential)
            hessian = torch.func.hessian(self.potential)
            g = gradient(x).detach()
            H = hessian(x).detach()
            lap = torch.func.grad(lambda z: hessian(z).diagonal().sum())(x).detach()
        checked(g,x,'gradient'); checked(lap,x,'gradient of Laplacian')
        if H.shape != (x.numel(),x.numel()) or not torch.isfinite(H).all():
            raise ValueError('invalid Hessian')
        return g, lambda v: H@v, torch.linalg.eigvalsh(H).abs().max(), lap

    def _one(self, x, z, za):
        h = self.step_size
        g,hvp,norm,lap = self._derivatives(x)
        hg = hvp(g)
        if self.tamed:
            gn,xn,ln = g.norm(),x.norm(),lap.norm()
            drift = g/(1+h**1.5*gn**1.5)**(2/3)
            correction = hg/(1+h*xn*norm*gn)
            lap = lap/(1+math.sqrt(h)*xn*ln)
            scale = 1+h*norm
        else:
            drift,correction,scale = g,hg,1.
        mean = x-h*drift+h*h/2*(correction-lap)
        return mean+math.sqrt(2*h)*(z-h/2*hvp(z)/scale+h/math.sqrt(12)*hvp(za)/scale)

    def step(self, *, noise=None, auxiliary_noise=None):
        x=self._position
        z=noise_like(x,self.generator,noise)
        za=noise_like(x,self.generator,auxiliary_noise)
        result=self._one(x,z,za) if x.ndim==1 else torch.stack([self._one(a,b,c) for a,b,c in zip(x,z,za)])
        return self._finish(result, step_size=self.step_size, tamed=self.tamed)
