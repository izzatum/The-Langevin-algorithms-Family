"""Simultaneous Euler Stein transports with bounded pair-interaction memory.

These deterministic interacting ensembles are posterior approximations, not
independent MCMC chains. No method retains a trajectory or a step autograd graph.
"""
import math
import torch
from lafa.targets import value_and_grad
from lafa.state import StepResult
from lafa.kernels import (RBF, validate_ensemble, validate_kernel,
    validate_block_size, normalized_weights, squared_distances, kernel_from_state)


def _validate_score(x, score):
    if not isinstance(score,torch.Tensor) or score.shape != x.shape or score.dtype != x.dtype or score.device != x.device or not torch.isfinite(score).all():
        raise ValueError('score must be finite and match particle shape, dtype and device')


def _identity(x):
    # This very tensor, its in-place version (inference tensors track none) and
    # the ambient inference mode, inside which autograd targets lose gradients.
    return (x,None if x.is_inference() else x._version,torch.is_inference_mode_enabled())


@torch.no_grad()
def direction(position, score, *, kernel=None, weights=None, block_size=128):
    """Liu--Wang Eq.(8): weighted source scores plus SOURCE kernel derivatives."""
    validate_ensemble(position)
    _validate_score(position,score)
    validate_block_size(block_size)
    kernel = RBF() if kernel is None else kernel
    validate_kernel(kernel)
    w = normalized_weights(position,weights)
    n,d = position.shape
    result = torch.zeros_like(position)
    for i in range(0,n,block_size):
        xi = position[i:i+block_size]
        for j in range(0,n,block_size):
            xj = position[j:j+block_size]
            k,b,_ = kernel.radial(squared_distances(xi,xj),d)
            kw,bw = k*w[j:j+block_size],b*w[j:j+block_size]
            result[i:i+block_size] += kw@score[j:j+block_size]
            # Direct per-destination differences preserve small separations even
            # when another particle is orders of magnitude farther away. Only
            # one (source block,D) coordinate workspace is live, never (B,B,D).
            for row in range(len(xi)):
                result[i+row] += bw[row]@(xi[row]-xj)
    if not torch.isfinite(result).all(): raise RuntimeError('nonfinite SVGD direction')
    return result


@torch.no_grad()
def kde_direction(position, score, *, kernel=None, weights=None, block_size=128):
    """KDE-WGD with own-particle attraction and normalized Gaussian KDE score.

    Log-sum-exp is accumulated across source tiles. Zero masses are permitted;
    no denominator epsilon or arbitrary repulsion coefficient is introduced.
    """
    validate_ensemble(position)
    _validate_score(position,score)
    validate_block_size(block_size)
    kernel = RBF() if kernel is None else kernel
    if not isinstance(kernel,RBF): raise ValueError('KDE-WGD requires a Gaussian RBF density kernel')
    w = normalized_weights(position,weights)
    logw = w.log()
    n,d = position.shape
    out = score.detach().clone()
    for i in range(0,n,block_size):
        xi = position[i:i+block_size]
        running_max = xi.new_full((len(xi),),-float('inf'))
        denominator = xi.new_zeros((len(xi),))
        repulsion_numerator = torch.zeros_like(xi)
        for j in range(0,n,block_size):
            xj = position[j:j+block_size]
            if not (w[j:j+block_size]>0).any(): continue
            logk = -.5*squared_distances(xi,xj)/kernel.length_scale**2+logw[j:j+block_size]
            new_max = torch.maximum(running_max,logk.max(-1).values)
            rescale = torch.exp(running_max-new_max)
            mass = torch.exp(logk-new_max[:,None])
            denominator = denominator*rescale+mass.sum(-1)
            repulsion_numerator *= rescale[:,None]
            for row in range(len(xi)):
                repulsion_numerator[row] += mass[row]@(xi[row]-xj)
            running_max = new_max
        out[i:i+block_size] += repulsion_numerator/denominator[:,None]/kernel.length_scale**2
    if not torch.isfinite(out).all(): raise RuntimeError('nonfinite KDE-WGD direction')
    return out


class SVGD:
    """Scalar-kernel parameter-space SVGD, Liu--Wang (2016), Eq.(8).

    Potential is independent-row batched negative log density; weights are
    fixed normalized empirical masses. block_size bounds both pair tiles and
    target evaluation batches, so target/model graphs are released between
    batches. Targets must not couple rows or change with batch composition.
    state_dict contains configuration and tensors, never the
    target closure. Restore into a constructed object with the target reattached.

    The evaluation of each accepted candidate is retained and reused by the next
    step instead of evaluating the same particles again; a target that is not
    deterministic is therefore drawn once per step. The retained evaluation is
    tied to that position tensor and to the target object, is dropped by
    load_state_dict and is never serialized. Call invalidate_cache() after
    changing target data/parameters in place; it also releases a replaced
    target, which otherwise stays referenced until the next step.
    """
    _cache = None  # (key, evaluation) of the current position.

    def __init__(self,potential,initial,*,step_size=.01,kernel=None,weights=None,block_size=128):
        validate_ensemble(initial)
        if not math.isfinite(step_size) or step_size <= 0: raise ValueError('step_size must be positive and finite')
        validate_block_size(block_size)
        self.kernel = RBF() if kernel is None else kernel
        validate_kernel(self.kernel)
        self.potential = potential
        self._position = initial.detach().clone()
        self.step_size = float(step_size)
        self.block_size = block_size
        self.weights = normalized_weights(initial,weights)
        self.iteration = 0

    @property
    def position(self):
        return self._position.detach().clone()

    def _batched_value_and_grad(self,potential,x):
        values=x.new_empty((len(x),))
        gradients=torch.empty_like(x)
        for start in range(0,len(x),self.block_size):
            batch=x[start:start+self.block_size]
            value,gradient=value_and_grad(potential,batch)
            if not getattr(potential,'score_only',False) and not torch.isfinite(value).all():
                raise ValueError('nonfinite potential values')
            _validate_score(batch,gradient)
            values[start:start+self.block_size]=value
            gradients[start:start+self.block_size]=gradient
        return values,gradients

    def _evaluate(self,x):
        return self._batched_value_and_grad(self.potential,x)

    def _record(self,x):
        """Everything retained for x: (values, gradient, *method extras)."""
        return self._evaluate(x)

    def _cache_key(self,x):
        # Whatever an evaluation of x reads; identity alone never proves equal content.
        return _identity(x)+(self.potential,self.block_size)

    def _retained(self,x):
        """Retained record of exactly this unmodified tensor and target, else None."""
        if self._cache is None: return None
        key,record = self._cache
        fresh = all(a is b or (type(a) is int and a == b) for a,b in zip(key,self._cache_key(x)))
        return record if fresh else None

    def invalidate_cache(self):
        """Drop the retained evaluation; the next step evaluates the current position.

        Call after changing target data/parameters in place. Nothing is evaluated here.
        """
        self._cache = None

    def __getstate__(self):
        # Copies and pickles carry no retained evaluation either.
        return {k:v for k,v in self.__dict__.items() if k != '_cache'}

    def _direction(self,x,gradient):
        return direction(x,-gradient,kernel=self.kernel,weights=self.weights,block_size=self.block_size),{}

    def step(self):
        x = self._position
        record = self._retained(x)
        if record is None:
            key = self._cache_key(x)
            record = self._record(x)
            self._cache = key,record
        field,stats = self._direction(x,record[1])
        candidate = (x+self.step_size*field).detach()
        validate_ensemble(candidate)
        key = self._cache_key(candidate)
        record = self._record(candidate)
        # A raise above leaves the position and its retained evaluation untouched.
        self._position,self._cache = candidate,(key,record)
        self.iteration += 1
        stats = {k:v.detach().clone() for k,v in stats.items()}
        stats['iteration'] = candidate.new_tensor(self.iteration,dtype=torch.int64)
        return StepResult(candidate.clone(),record[0].clone(),record[1].clone(),stats=stats)

    def state_dict(self):
        return {'version':1,'method':type(self).__name__,'position':self.position,
                'iteration':self.iteration,'step_size':self.step_size,
                'block_size':self.block_size,'weights':self.weights.clone(),
                'kernel':self.kernel.state_dict(),'extra':self._extra_state(),
                'reattach':'target/function callbacks are not serialized'}

    def _extra_state(self): return {}

    def load_state_dict(self,state):
        if state.get('version') != 1 or state.get('method') != type(self).__name__:
            raise ValueError('checkpoint version/method mismatch')
        x = state['position'].to(device=self._position.device,dtype=self._position.dtype).detach().clone()
        validate_ensemble(x)
        if x.shape != self._position.shape: raise ValueError('checkpoint ensemble shape mismatch')
        step = state['step_size']
        if not math.isfinite(step) or step <= 0: raise ValueError('invalid checkpoint step_size')
        validate_block_size(state['block_size'])
        if isinstance(state['iteration'],bool) or not isinstance(state['iteration'],int) or state['iteration']<0:
            raise ValueError('invalid checkpoint iteration')
        kernel = kernel_from_state(state['kernel'])
        w = torch.as_tensor(state['weights'],device=x.device,dtype=x.dtype).detach().clone()
        normalized_weights(x,w)  # Validate; retain exact saved masses for continuation.
        if not torch.allclose(w.sum(),x.new_tensor(1.)): raise ValueError('checkpoint masses must sum to one')
        self._load_extra(state['extra'],x)
        self._position,self.iteration,self.step_size,self.block_size = x,state['iteration'],step,state['block_size']
        self.kernel,self.weights = kernel,w
        self._cache = None  # A restored position owns no evaluation.
        return self

    def _load_extra(self,extra,x):
        if extra: raise ValueError('unexpected checkpoint metadata')


# Weight-space SVGD is the established parameter-space algorithm, not a new method.
WeightSpaceSVGD = SVGD
