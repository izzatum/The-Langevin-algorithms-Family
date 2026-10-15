"""Stationary radial kernels with explicit, frozen scalar hyperparameters.

RBF uses exp(-distance²/(2*length_scale²)), not exp(-distance²/h).
Pairwise routines allocate scalar tiles, never a pair-by-coordinate tensor.
"""
from dataclasses import dataclass, asdict
import math
import torch
from lafa.targets import validate_position


def validate_ensemble(x):
    validate_position(x)
    if x.ndim != 2:
        raise ValueError('particles must have shape (N,D)')


def validate_block_size(block_size):
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer')


def normalized_weights(x, weights=None):
    if weights is None:
        return x.new_full((len(x),), 1/len(x))
    w = torch.as_tensor(weights, device=x.device, dtype=x.dtype).detach().clone()
    if w.shape != (len(x),) or not torch.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
        raise ValueError('weights must be finite nonnegative masses with positive sum and shape (N,)')
    # Scaling first avoids overflowing the sum of valid large weights.
    w /= w.max()
    return w/w.sum()


def squared_distances(x, y):
    """Stable scalar distance tile; direct cdist avoids norm cancellation.

    PyTorch's direct distance kernel reduces coordinates without materializing
    an (N,M,D) difference tensor. Tile dimensions are controlled by callers.
    """
    t = torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist').square()
    if not torch.isfinite(t).all():
        raise ValueError('pairwise squared distances overflowed; rescale coordinates or use float64')
    return t


@dataclass(frozen=True)
class RBF:
    length_scale: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError('length_scale must be finite and positive')

    def radial(self, t, dimension):
        inv = 1/self.length_scale**2
        k = torch.exp(-.5*inv*t)
        return k, k*inv, k*(dimension*inv-t*inv**2)

    def __call__(self, x, y):
        return self.radial(squared_distances(x,y),x.shape[-1])[0]

    def state_dict(self):
        return dict(kind='rbf', **asdict(self))


@dataclass(frozen=True)
class IMQ:
    scale: float = 1.0
    beta: float = -0.5

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('scale must be finite and positive')
        if not math.isfinite(self.beta) or not -1 < self.beta < 0:
            raise ValueError('beta must lie strictly between -1 and 0')

    def radial(self,t,dimension):
        base = self.scale**2+t
        fprime = self.beta*base.pow(self.beta-1)
        fsecond = self.beta*(self.beta-1)*base.pow(self.beta-2)
        return base.pow(self.beta), -2*fprime, -2*dimension*fprime-4*t*fsecond

    def __call__(self,x,y):
        return self.radial(squared_distances(x,y),x.shape[-1])[0]

    def state_dict(self):
        return dict(kind='imq', **asdict(self))


def kernel_from_state(state):
    data = dict(state)
    kind = data.pop('kind')
    if kind == 'rbf': return RBF(**data)
    if kind == 'imq': return IMQ(**data)
    raise ValueError('unknown kernel kind')


def validate_kernel(kernel):
    if not isinstance(kernel,(RBF,IMQ)):
        raise TypeError('kernel must be RBF or IMQ')


def median_rbf(samples, *, max_pairs=4096, seed=0, squared_distance_floor=1e-8):
    """Opt-in approximate median heuristic, deterministic for seed/device.

    Sample at most max_pairs ordered off-diagonal pairs WITH replacement;
    h=max(median(squared distances),floor)/log(max(N,2)), ell=sqrt(h/2).
    Selection and result are detached; no bandwidth derivative enters dynamics.
    This bounded sample is approximate even when max_pairs exceeds N*(N-1).
    """
    validate_ensemble(samples)
    validate_block_size(max_pairs)
    if not math.isfinite(squared_distance_floor) or squared_distance_floor <= 0:
        raise ValueError('squared_distance_floor must be positive and finite')
    n = len(samples)
    median = squared_distance_floor
    if n > 1:
        rng = torch.Generator(device=samples.device).manual_seed(seed)
        count = min(max_pairs,n*(n-1))
        i = torch.randint(n,(count,),device=samples.device,generator=rng)
        j = torch.randint(n-1,(count,),device=samples.device,generator=rng)
        j = j+(j>=i)
        with torch.no_grad():
            distances = samples.new_empty((count,))
            for offset in range(0,count,128):
                distances[offset:offset+128] = (samples[i[offset:offset+128]]-samples[j[offset:offset+128]]).square().sum(-1)
            if not torch.isfinite(distances).all(): raise ValueError('median distances overflowed')
            median = max(float(distances.median()),squared_distance_floor)
    return RBF(math.sqrt(median/(2*math.log(max(n,2)))))

__all__ = ['RBF','IMQ','median_rbf']
