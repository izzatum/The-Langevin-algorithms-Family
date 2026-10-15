"""Independent benchmark reference: elliptical slice sampling for Gaussian priors.

Murray, Adams and MacKay (2010), *Elliptical Slice Sampling*, AISTATS 9,
541–548, https://proceedings.mlr.press/v9/murray10a.html. This implementation
uses no Langevin transitions, gradients, or candidate sampler internals. Finite
runs still require convergence checks; invariance does not establish mixing.
"""
import math
import time
import torch


@torch.no_grad()
def elliptical_slice(log_likelihood, prior_mean, prior_cholesky, *, seeds,
                     draws, warmup=500, initial=None, max_shrinks=10000):
    """Sample independent chains, returning owned ``(chains, draws, dims)``.

    ``log_likelihood`` accepts a chain batch and returns one scalar per chain.
    Each chain has its own explicitly seeded CPU generator; the Gaussian prior
    is N(prior_mean, prior_cholesky @ prior_cholesky.T). The slice bracket spans
    the full ellipse and shrinks around angle zero. A shrink cap raises rather
    than silently returning a biased transition. Timing includes warmup.
    """
    mean = torch.as_tensor(prior_mean)
    chol = torch.as_tensor(prior_cholesky, dtype=mean.dtype, device=mean.device)
    seeds = tuple(seeds)
    if mean.ndim != 1 or not mean.is_floating_point() or mean.device.type != 'cpu':
        raise ValueError('reference requires a floating CPU prior mean vector')
    d, c = mean.numel(), len(seeds)
    if chol.shape != (d, d) or not torch.isfinite(chol).all() or not torch.isfinite(mean).all():
        raise ValueError('finite matching prior parameters required')
    if not torch.equal(chol, torch.tril(chol)) or (chol.diag() <= 0).any():
        raise ValueError('prior_cholesky must be lower triangular with positive diagonal')
    if not c or len(set(seeds)) != c:
        raise ValueError('provide distinct independent chain seeds')
    if any(not isinstance(n, int) for n in (draws, warmup, max_shrinks)) or draws < 1 or warmup < 0 or max_shrinks < 1:
        raise ValueError('positive integer draws/max_shrinks and nonnegative warmup required')
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]

    def normal():
        return torch.stack([torch.randn(d, generator=g, dtype=mean.dtype) for g in generators])

    def uniform():
        return torch.stack([torch.rand((), generator=g, dtype=mean.dtype) for g in generators])

    started = time.perf_counter()
    x = mean + normal() @ chol.T if initial is None else torch.as_tensor(initial, dtype=mean.dtype).clone()
    if x.shape != (c, d) or not torch.isfinite(x).all():
        raise ValueError('initial must be finite with shape (chains, dimensions)')
    ll = log_likelihood(x)
    if ll.shape != (c,) or not torch.isfinite(ll).all():
        raise ValueError('initial log likelihood must be finite with one value per chain')
    evaluations = c
    samples = torch.empty(c, draws, d, dtype=mean.dtype)
    elapsed = torch.empty(draws, dtype=torch.float64)
    shrink_counts = []
    for step in range(warmup + draws):
        direction = normal() @ chol.T
        log_height = ll + uniform().log()
        theta = 2 * math.pi * uniform()
        low, high = theta - 2 * math.pi, theta.clone()
        pending = torch.ones(c, dtype=torch.bool)
        centered = x - mean
        for shrink in range(max_shrinks):
            proposal = mean + centered * theta.cos()[:, None] + direction * theta.sin()[:, None]
            proposed_ll = log_likelihood(proposal)
            evaluations += c  # Actual batched work includes already accepted rows.
            accept = pending & torch.isfinite(proposed_ll) & (proposed_ll >= log_height)
            x = torch.where(accept[:, None], proposal, x)
            ll = torch.where(accept, proposed_ll, ll)
            pending &= ~accept
            if not pending.any():
                break
            low = torch.where(pending & (theta < 0), theta, low)
            high = torch.where(pending & (theta >= 0), theta, high)
            theta = low + uniform() * (high - low)
        else:
            raise RuntimeError('elliptical slice bracket exceeded max_shrinks')
        shrink_counts.append(shrink)
        if step >= warmup:
            samples[:, step - warmup] = x
            elapsed[step - warmup] = time.perf_counter() - started
    return dict(samples=samples, elapsed_seconds=elapsed, seconds=time.perf_counter()-started,
                likelihood_chain_evaluations=evaluations, seeds=list(seeds), warmup=warmup,
                draws=draws, max_bracket_shrinks=max(shrink_counts))
