"""Temperature annealing, noise-score annealing and corrected replica exchange."""
import math
import torch
from ..targets import value_and_grad
from ._common import OwnedState, checked, positive, noise_like


class ScheduleComplete(RuntimeError):
    """step() was called after the last scheduled stage.

    Not StopIteration: Python rewrites that into RuntimeError inside a generator
    and iterator protocols swallow it. Poll `finished` to stop without exceptions.
    """


def _schedule(values,name):
    result=tuple(positive(v,name) for v in values)
    if not result:
        raise ValueError(f'{name} must be nonempty')
    return result


def _count(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class AnnealedLangevin(OwnedState):
    """Known-potential temperature annealing (distinct from R08 score annealing).

    x <- x-h grad U + sqrt(2hT)z. Each stage targets exp(-U/T) up to ULA bias;
    changing T defines a nonstationary process. `finished` turns True when the
    last scheduled step completes; stepping past it raises ScheduleComplete.
    No claim that the endpoint is an exact final-target draw.
    """
    _state_fields=OwnedState._state_fields+('temperatures','steps_per_temperature','stage','stage_iteration')
    def __init__(self,potential,initial,temperatures,step_size=.01,*,steps_per_temperature=100,generator=None):
        self._initialize(initial,generator)
        self.potential=potential
        self.step_size=positive(step_size,'step_size')
        self.temperatures=_schedule(temperatures,'temperature')
        self.steps_per_temperature=_count(steps_per_temperature,'steps_per_temperature')
        self.stage=self.stage_iteration=0

    @property
    def finished(self):
        return self.stage>=len(self.temperatures)

    def step(self,*,noise=None):
        if self.finished:
            raise ScheduleComplete('temperature schedule complete')
        t=self.temperatures[self.stage]
        _,g=value_and_grad(self.potential,self._position)
        result=self._finish(self._position-self.step_size*g+math.sqrt(2*self.step_size*t)*noise_like(self._position,self.generator,noise),temperature=t,stage=self.stage)
        self.stage_iteration+=1
        if self.stage_iteration==self.steps_per_temperature:
            self.stage+=1; self.stage_iteration=0
        return result


class NoiseAnnealedLangevin(OwnedState):
    """R08 Algorithm 1: supplied noise-conditional score, no model training.

    alpha_i=epsilon*sigma_i²/sigma_last², x+=alpha_i*score(x,sigma_i)/2
    +sqrt(alpha_i)z (source half-drift convention). Scores approximate gradients
    of log noise-smoothed densities. Finite schedules/endpoints are approximate.
    `finished` turns True when the last scheduled step completes; stepping past
    it raises ScheduleComplete.
    """
    _state_fields=OwnedState._state_fields+('sigmas','steps_per_sigma','stage','stage_iteration')
    def __init__(self,score,initial,sigmas,step_size=2e-5,*,steps_per_sigma=100,generator=None):
        self._initialize(initial,generator)
        self.score=score
        self.step_size=positive(step_size,'step_size')
        self.sigmas=_schedule(sigmas,'sigma')
        if any(a<b for a,b in zip(self.sigmas,self.sigmas[1:])):
            raise ValueError('sigmas must be nonincreasing')
        self.steps_per_sigma=_count(steps_per_sigma,'steps_per_sigma')
        self.stage=self.stage_iteration=0

    @property
    def finished(self):
        return self.stage>=len(self.sigmas)

    def step(self,*,noise=None):
        if self.finished:
            raise ScheduleComplete('noise schedule complete')
        sigma=self.sigmas[self.stage]
        alpha=self.step_size*(sigma/self.sigmas[-1])**2
        with torch.no_grad():
            score=checked(self.score(self._position.clone(),sigma),self._position,'score')
        result=self._finish(self._position+alpha/2*score+math.sqrt(alpha)*noise_like(self._position,self.generator,noise),sigma=sigma,alpha=alpha,stage=self.stage)
        self.stage_iteration+=1
        if self.stage_iteration==self.steps_per_sigma:
            self.stage+=1; self.stage_iteration=0
        return result


class ReplicaExchange(OwnedState):
    """R09/R19: fixed-temperature MALA replicas with alternating neighbor swaps.

    initial is (R,D), temperatures strictly increasing. Local kernels use
    x-h grad U+sqrt(2hT)z and full MALA correction. Swaps use untempered energies
    and (1/T_i-1/T_j)*(U_i-U_j); fixed product target is invariant under standard
    MALA conditions. Report cold replica separately; no adaptive temperatures.

    The current (U,grad U) is carried between steps, so K steps cost K+1 target
    batches; this assumes a deterministic potential. The pair is never
    serialised: load_state_dict drops it and the next step re-evaluates.
    Assigning a new `potential` object is detected; after changing its data or
    parameters in place call invalidate_cache(). A carried row was evaluated
    before its swap, at another batch position, so agreement with a
    re-evaluating run, and hence with a run resumed from a checkpoint, is
    bit-exact only when a row's value and gradient do not depend on its batch
    position; otherwise it holds to rounding. invalidate_cache() before every
    step restores re-evaluation at 2K+1 batches.
    """
    _state_fields=OwnedState._state_fields+('temperatures',)
    _cache=None
    def __init__(self,potential,initial,temperatures,step_size=.01,*,generator=None):
        self._initialize(initial,generator)
        self.potential=potential
        self.step_size=positive(step_size,'step_size')
        ts=_schedule(temperatures,'temperature')
        if initial.ndim!=2 or len(ts)!=len(initial) or len(ts)<2 or any(a>=b for a,b in zip(ts,ts[1:])):
            raise ValueError('one strictly increasing temperature per replica, R>=2')
        self.temperatures=initial.new_tensor(ts)
        v,g=value_and_grad(potential,initial)
        self._cache=(potential,checked(v,initial,'initial potential',(len(initial),)),checked(g,initial,'initial gradient'))

    @property
    def cold_position(self):
        return self._position[0].clone()

    def invalidate_cache(self):
        """Forget the carried (U,grad U); the next step re-evaluates the current replicas."""
        self._cache=None

    def load_state_dict(self,checkpoint):
        self.invalidate_cache()
        super().load_state_dict(checkpoint)

    def _current(self):
        if self._cache is None or self._cache[0] is not self.potential:
            self._cache=(self.potential,*value_and_grad(self.potential,self._position))
        return self._cache[1:]

    def step(self,*,noise=None,local_uniforms=None,swap_uniforms=None):
        x=self._position
        h=self.step_size
        v,g=self._current()
        proposal=x-h*g+(2*h*self.temperatures).sqrt()[:,None]*noise_like(x,self.generator,noise)
        vn,gn=value_and_grad(self.potential,proposal)
        forward=(proposal-x+h*g).square().sum(-1)
        backward=(x-proposal+h*gn).square().sum(-1)
        ratio=(v-vn)/self.temperatures+(forward-backward)/(4*h*self.temperatures)
        uniforms=torch.rand(len(x),device=x.device,dtype=x.dtype,generator=self.generator) if local_uniforms is None else checked(local_uniforms,x,'local uniforms',(len(x),))
        if ((uniforms<0)|(uniforms>1)).any():
            raise ValueError('uniforms must be in [0,1]')
        accepted=torch.isfinite(ratio)&torch.isfinite(gn).all(-1)&torch.isfinite(proposal).all(-1)&(uniforms.log()<ratio.clamp_max(0))
        x=torch.where(accepted[:,None],proposal,x).detach()
        v=torch.where(accepted,vn,v).detach()
        g=torch.where(accepted[:,None],gn,g).detach()
        pairs=list(range(self.iteration%2,len(x)-1,2)) if len(x)>2 else [0]
        uniforms=torch.rand(len(pairs),device=x.device,dtype=x.dtype,generator=self.generator) if swap_uniforms is None else checked(swap_uniforms,x,'swap uniforms',(len(pairs),))
        if ((uniforms<0)|(uniforms>1)).any():
            raise ValueError('uniforms must be in [0,1]')
        swapped=torch.zeros(len(x)-1,device=x.device,dtype=torch.bool)
        for k,i in enumerate(pairs):
            j=i+1
            log_a=(1/self.temperatures[i]-1/self.temperatures[j])*(v[i]-v[j])
            if uniforms[k].log()<log_a.clamp_max(0):
                x[[i,j]]=x[[j,i]].clone()
                v[[i,j]]=v[[j,i]].clone()
                g[[i,j]]=g[[j,i]].clone()
                swapped[i]=True
        # Dropped across the position commit: an interrupt in between must not
        # leave the previous step's pair attached to the new replicas.
        self._cache=None
        result=self._finish(x,local_accepted=accepted,swapped=swapped,temperatures=self.temperatures)
        self._cache=(self.potential,v,g)
        return result
