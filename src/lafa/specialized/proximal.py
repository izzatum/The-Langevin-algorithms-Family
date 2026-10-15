"""Shared-parameter proximal interacting particle algorithms (R10)."""
import math
import torch
from ._common import OwnedState, InteractingResult, checked, positive, noise_like
from ..targets import validate_position


class _PIP(OwnedState):
    _state_fields = OwnedState._state_fields + ('_theta', 'smoothing')

    def _setup(self, smooth_potential, prox_joint, theta, particles, step_size, smoothing, generator):
        validate_position(theta)
        if theta.ndim != 1 or particles.ndim != 2:
            raise ValueError('theta must be (P,) and particles (N,D)')
        self._initialize(particles,generator)
        if theta.dtype != particles.dtype or theta.device != particles.device:
            raise ValueError('theta and particles must share dtype/device')
        self._theta=theta.detach().clone()
        self.smooth_potential=smooth_potential
        self.prox_joint=prox_joint
        self.step_size=positive(step_size,'step_size')
        self.smoothing=positive(smoothing,'smoothing')

    @property
    def theta(self):
        return self._theta.clone()

    @property
    def particles(self):
        return self.position

    def _gradients(self):
        with torch.enable_grad():
            t=self._theta.detach().clone().requires_grad_()
            x=self._position.detach().clone().requires_grad_()
            # Nonleaf copies let callbacks use in-place operations while the
            # original leaves remain the differentiation coordinates.
            values=self.smooth_potential(t.clone(),x.clone())
            checked(values,x,'smooth potential',(x.shape[0],))
            gs=torch.autograd.grad(values.sum(),(t,x),allow_unused=True) if values.requires_grad else (None,None)
        gt=torch.zeros_like(t) if gs[0] is None else gs[0]/x.shape[0]
        gx=torch.zeros_like(x) if gs[1] is None else gs[1]
        return checked(gt,t,'theta gradient'),checked(gx,x,'latent gradient')

    def _prox(self,t,x):
        # Inputs are private work buffers; callbacks may consume them.
        with torch.no_grad():
            pt,px=self.prox_joint(t,x,self.smoothing)
        pt=checked(pt,t,'per-particle parameter prox',(x.shape[0],t.numel()))
        px=checked(px,x,'latent prox')
        return pt.mean(0),px

    def _finish_pair(self,t,x):
        t=checked(t,self._theta,'theta transition')
        self._finish(x)
        self._theta=t
        return InteractingResult(self.theta,self.particles,{'iteration':self.iteration,'smoothing':self.smoothing})

    def load_state_dict(self,checkpoint):
        checked(checkpoint['state']['_theta'],self._theta,'checkpoint theta')
        super().load_state_dict(checkpoint)


class MYIPLA(_PIP):
    """R10 Eqs. (9)-(10): Moreau-Yosida interacting Langevin, finite-step biased.

    smooth_potential(theta, particles) -> (N,) separates latent particles.
    prox_joint(theta,particles,lambda) -> ((N,P),(N,D)) is the exact *joint*
    prox of g2 for each (theta,x_i). One shared theta learns the powered marginal;
    theta samples are not an ordinary Bayesian parameter posterior.
    """
    def __init__(self,smooth_potential,prox_joint,theta,particles,step_size=.01,*,smoothing=.1,generator=None):
        self._setup(smooth_potential,prox_joint,theta,particles,step_size,smoothing,generator)

    def step(self,*,theta_noise=None,particle_noise=None):
        h,l=self.step_size,self.smoothing
        gt,gx=self._gradients()
        pt,px=self._prox(self._theta.clone(),self._position.clone())
        t=self._theta-h*gt-h/l*(self._theta-pt)+math.sqrt(2*h/self._position.shape[0])*noise_like(self._theta,self.generator,theta_noise)
        x=self._position-h*gx-h/l*(self._position-px)+math.sqrt(2*h)*noise_like(self._position,self.generator,particle_noise)
        return self._finish_pair(t,x)


class PIPULA(_PIP):
    """R10 section 3.1.1: g1=0, g2=full U and lambda=h; noise follows prox.

    prox_joint(theta,particles,h) returns ((N,P),(N,D)), the exact full joint
    model prox. There is no additional smooth gradient. Noise may leave hard
    support constraints. Shared theta has noise covariance 2h/N.
    """
    def __init__(self,prox_joint,theta,particles,step_size=.01,*,generator=None):
        self._setup(None,prox_joint,theta,particles,step_size,step_size,generator)

    def step(self,*,theta_noise=None,particle_noise=None):
        h=self.step_size
        t,x=self._prox(self._theta.clone(),self._position.clone())
        t=t+math.sqrt(2*h/self._position.shape[0])*noise_like(self._theta,self.generator,theta_noise)
        x=x+math.sqrt(2*h)*noise_like(self._position,self.generator,particle_noise)
        return self._finish_pair(t,x)


class PIPGLA(_PIP):
    """R10 Eqs. (11)-(14): interacting smooth ULA, then per-particle joint prox.

    Same model/prox contract as MYIPLA. smoothing defaults to h. Distinct h and
    lambda are allowed but need not meet the paper's convexity/rate hypotheses.
    Projection occurs after noise; averaging theta can affect coupled feasibility.
    """
    def __init__(self,smooth_potential,prox_joint,theta,particles,step_size=.01,*,smoothing=None,generator=None):
        self._setup(smooth_potential,prox_joint,theta,particles,step_size,step_size if smoothing is None else smoothing,generator)

    def step(self,*,theta_noise=None,particle_noise=None):
        h=self.step_size
        gt,gx=self._gradients()
        t=self._theta-h*gt+math.sqrt(2*h/self._position.shape[0])*noise_like(self._theta,self.generator,theta_noise)
        x=self._position-h*gx+math.sqrt(2*h)*noise_like(self._position,self.generator,particle_noise)
        t,x=self._prox(t,x)
        return self._finish_pair(t,x)
