"""Empirical Langevin Stein discrepancy, with explicit estimator conventions."""
import torch
from lafa.kernels import (RBF, validate_ensemble, validate_kernel,
    validate_block_size, normalized_weights, squared_distances)


@torch.no_grad()
def ksd(samples, score, *, kernel=None, estimator='v', weights=None,
        block_size=128, squared=True):
    """Return exact tiled squared KSD (V) or signed off-diagonal estimate (U).

    score is an (N,D) tensor or independent-row callable returning grad log pi
    (not grad U). Callable evaluation is batched at block_size and detached
    before the next batch. V includes diagonals and is the empirical-measure
    discrepancy. U divides by sum(i!=j,w_i*w_j), mathematically 1-sum(w²),
    computed as positive pair masses to avoid dominant-weight cancellation.
    Unbiasedness needs IID draws and sample-independent masses.
    Negative U is retained, and squared=False is disallowed for U. V square
    roots tolerate only roundoff negativity (64*machine epsilon*absolute sum).
    Small RBF KSD is not a universal convergence certificate. IMQ guarantees
    likewise need target regularity (Gorham--Mackey, PMLR 70, Proposition 2).
    Work memory is O(ND+block_size²); all ordered particle pairs are retained.
    """
    validate_ensemble(samples)
    validate_block_size(block_size)
    kernel = RBF() if kernel is None else kernel
    validate_kernel(kernel)
    if estimator not in ('u','v'): raise ValueError("estimator must be 'u' or 'v'")
    if estimator == 'u' and not squared:
        raise ValueError('U is a signed squared-discrepancy estimator; cannot take its square root')
    if callable(score):
        callback=score
        score=torch.empty_like(samples)
        for start in range(0,len(samples),block_size):
            batch=samples[start:start+block_size].detach()
            with torch.enable_grad(): evaluated=callback(batch)
            if not isinstance(evaluated,torch.Tensor) or evaluated.shape!=batch.shape or evaluated.dtype!=batch.dtype or evaluated.device!=batch.device:
                raise ValueError('score callback must preserve batch shape, dtype and device')
            score[start:start+block_size]=evaluated.detach()
            del evaluated
    if not isinstance(score,torch.Tensor) or score.shape != samples.shape or score.device != samples.device or score.dtype != samples.dtype or not torch.isfinite(score).all():
        raise ValueError('score must be finite and match samples shape, dtype and device')
    score = score.detach()
    # Double accumulation also protects tiny nonzero off-diagonal mass products
    # for float32 inputs. Kernel arithmetic and returned scalar retain input dtype.
    w = normalized_weights(samples,weights).to(torch.float64)
    denominator = 2*(w[1:]*w[:-1].cumsum(0)).sum()
    if estimator == 'u' and not denominator > 0:
        raise ValueError('U requires at least two positive masses')
    total, magnitude = w.new_zeros(()),w.new_zeros(())
    n,d = samples.shape
    for i in range(0,n,block_size):
        xi,si,wi = samples[i:i+block_size],score[i:i+block_size],w[i:i+block_size]
        for j in range(0,n,block_size):
            xj,sj,wj = samples[j:j+block_size],score[j:j+block_size],w[j:j+block_size]
            k,b,mixed = kernel.radial(squared_distances(xi,xj),d)
            # Direct per-row residual products retain nearby separations even
            # in mixed-scale tiles, using only one (source block,D) workspace.
            cross=torch.empty_like(k)
            for row in range(len(xi)):
                cross[row]=((xi[row]-xj)*(si[row]-sj)).sum(-1)
            terms = (k*(si@sj.T)+b*cross+mixed)*(wi[:,None]*wj[None,:])
            if estimator == 'u' and i==j: terms.diagonal().zero_()
            total += terms.sum()
            magnitude += terms.abs().sum()
    if not torch.isfinite(total): raise RuntimeError('nonfinite Stein discrepancy; rescale or use float64')
    if estimator == 'u': return (total/denominator).to(samples.dtype)
    tolerance = 64*torch.finfo(samples.dtype).eps*magnitude.clamp_min(1)
    if total < -tolerance: raise RuntimeError('materially negative V-statistic indicates numerical failure')
    return (total if squared else total.clamp_min(0).sqrt()).to(samples.dtype)

__all__ = ['ksd']
