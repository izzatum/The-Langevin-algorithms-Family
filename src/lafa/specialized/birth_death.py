"""Fixed-population birth/death accelerated Langevin, R18 Algorithm 1."""
import math
import torch
from ..targets import value_and_grad
from ._common import OwnedState, checked, positive, noise_like


def gaussian_log_kde(particles, bandwidth, *, block_size=128):
    """Exact Gaussian KDE including self terms, with bounded pair blocks."""
    b=positive(bandwidth,'bandwidth')
    if particles.ndim!=2 or block_size<1:
        raise ValueError('particles must be (N,D), block_size positive')
    n,d=particles.shape
    result=[]
    for a in particles.split(block_size):
        accum=a.new_full((len(a),),-float('inf'))
        for other in particles.split(block_size):
            squared=(a[:,None,:]-other[None,:,:]).square().sum(-1)
            accum=torch.logaddexp(accum,torch.logsumexp(-squared/(2*b*b),dim=1))
        result.append(accum-math.log(n)-d*math.log(b*math.sqrt(2*math.pi)))
    return torch.cat(result)


class BirthDeathLangevin(OwnedState):
    """R18: equal-weight BDLS, synchronous ULA then sequential reactions.

    Centered KDE+U rates are recomputed before EVERY event candidate. Positive
    rate kills the slot, negative rate duplicates it into a uniformly chosen
    OTHER slot. This explicitly resolves the source's ambiguous sweep ordering.
    Exact blockwise KDE avoids an N² matrix; O(N³d) reference sweep is expensive.
    Finite N, bandwidth and step induce approximation; no finite-particle exactness.
    reaction_strength=0 is a diagnostic transport-only extension.
    """
    _state_fields=OwnedState._state_fields+('bandwidth','block_size','reaction_strength','_ancestry')

    def __init__(self,potential,initial,step_size=.01,*,bandwidth=1.,block_size=128,reaction_strength=1.,generator=None):
        self._initialize(initial,generator)
        if initial.ndim!=2 or initial.shape[0]<2:
            raise ValueError('BDLS requires particles (N,D), N>=2')
        self.potential=potential
        self.step_size=positive(step_size,'step_size')
        self.bandwidth=positive(bandwidth,'bandwidth')
        self.block_size=int(block_size)
        if self.block_size<1:
            raise ValueError('block_size must be positive')
        self.reaction_strength=float(reaction_strength)
        if not math.isfinite(self.reaction_strength) or self.reaction_strength<0:
            raise ValueError('reaction_strength must be finite and nonnegative')
        self._ancestry=torch.arange(len(initial),device=initial.device)

    @property
    def particles(self):
        return self.position

    def rates(self,particles=None):
        x=self._position if particles is None else particles
        with torch.no_grad():
            u=self.potential(x.detach().clone())
            u=checked(u,x,'potential',(len(x),))
            beta=gaussian_log_kde(x,self.bandwidth,block_size=self.block_size)+u
        return (beta-beta.mean()).detach()

    def step(self,*,noise=None,uniforms=None,partners=None,order=None):
        x=self._position
        n=len(x)
        _,g=value_and_grad(self.potential,x)
        x=checked(x-self.step_size*g+math.sqrt(2*self.step_size)*noise_like(x,self.generator,noise),x,'transport')
        uniforms=torch.rand(n,device=x.device,dtype=x.dtype,generator=self.generator) if uniforms is None else checked(uniforms,x,'uniforms',(n,))
        if ((uniforms<0)|(uniforms>1)).any():
            raise ValueError('uniforms must lie in [0,1]')
        if order is None:
            order=torch.arange(n,device=x.device)
        if order.shape!=(n,) or order.dtype!=torch.long or sorted(order.tolist())!=list(range(n)):
            raise ValueError('order must be an integer permutation')
        if partners is None:
            raw=torch.randint(n-1,(n,),device=x.device,generator=self.generator)
            slots=torch.arange(n,device=x.device)
            partners=raw+(raw>=slots)
        if partners.shape!=(n,) or partners.dtype!=torch.long or ((partners<0)|(partners>=n)|(partners==torch.arange(n,device=partners.device))).any():
            raise ValueError('partners must choose another valid slot for each slot')
        ancestry=torch.arange(n,device=x.device)
        deaths=births=0
        for index in order.tolist():
            rate=self.rates(x)[index]
            probability=-torch.expm1(-self.step_size*self.reaction_strength*rate.abs())
            if uniforms[index]<probability:
                partner=int(partners[index])
                dest,source=(index,partner) if rate>0 else (partner,index)
                x[dest]=x[source].clone()
                ancestry[dest]=ancestry[source].clone()
                deaths+=int(rate>0)
                births+=int(rate<0)
        result=self._finish(x,ancestry=ancestry,births=births,deaths=deaths,weights=x.new_full((n,),1/n))
        self._ancestry=self._ancestry[ancestry].clone()
        return result
