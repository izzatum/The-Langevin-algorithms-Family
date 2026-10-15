"""Diagnostics for INDEPENDENT stationary chains, never interacting particles.

Inputs are (chains, draws, dimensions), with adaptation and warmup removed.
Rhat and split quality diagnostics follow Vehtari et al. (2021), Bayesian Analysis 16(2), rank normalization,
folding, and splitting. ESS uses a pooled variance estimate and Geyer's initial
positive, monotone autocorrelation pairs. The original
effective_sample_size API is raw-parameter mean ESS, capped at total draws.
Separate bulk/tail ESS and mean/quantile MCSE functions use split draws and
require at least ten retained draws per independent chain.
"""
import torch


def _validate_chains(chains, *, min_chains=1, min_draws=2):
    if not isinstance(chains,torch.Tensor) or chains.dtype not in (torch.float32,torch.float64):
        raise ValueError('chains must be a float32/float64 tensor')
    if chains.ndim!=3 or chains.shape[0]<min_chains or chains.shape[1]<min_draws or chains.shape[2]<1 or not torch.isfinite(chains).all():
        raise ValueError(f'expected independent stationary chains with shape (chains>={min_chains}, draws>={min_draws}, dimensions>=1)')


def _scaled_chains(chains):
    # One positive scale per dimension across ALL chains preserves relative
    # within-/between-chain variability. Scaling before means, variances and
    # FFT products prevents both overflow and underflow of squared units.
    scale=chains.abs().amax(dim=(0,1),keepdim=True)
    return chains/torch.where(scale>0,scale,torch.ones_like(scale))


def _autocovariance(chains):
    n=chains.shape[1]
    centered=chains-chains.mean(1,keepdim=True)
    nfft=1 << (2*n-1).bit_length()
    transform=torch.fft.rfft(centered,n=nfft,dim=1)
    acov=torch.fft.irfft(transform*transform.conj(),n=nfft,dim=1)[:,:n]/n
    if not torch.isfinite(acov).all():
        raise RuntimeError('nonfinite autocovariance; numerical chain diagnostics are undefined')
    return acov


@torch.no_grad()
def autocorrelation(chains, *, max_lag=None):
    """Biased FFT autocorrelation, per chain/dimension; constant draws give NaN.

    Return shape (chains,max_lag+1,dimensions), including lag zero. The biased
    covariance divisor is n at every lag; this is NOT the n-lag normalization.
    A common per-dimension scaling precedes FFT products to avoid overflow.
    """
    _validate_chains(chains)
    if max_lag is None: max_lag=chains.shape[1]-1
    if isinstance(max_lag,bool) or not isinstance(max_lag,int) or not 0<=max_lag<chains.shape[1]:
        raise ValueError('max_lag must be an integer in [0,draws-1]')
    acov=_autocovariance(_scaled_chains(chains))
    return acov[:,:max_lag+1]/acov[:,:1]


@torch.no_grad()
def effective_sample_size(chains):
    """Raw mean ESS via pooled-chain Geyer positive/monotone sequence.

    This finite-run estimate assumes stationary, independently initialized
    Markov chains, not particles, optimization iterates, or warmup. Between-chain
    disagreement inflates the variance estimate. Return one ESS per dimension;
    globally constant draws give NaN. Antithetic ESS is capped at total draws,
    so this is deliberately conservative compared with uncapped alternatives.
    A common per-dimension scaling precedes variance and FFT operations; invalid
    intermediate correlations yield NaN, never a manufactured maximum ESS.
    """
    _validate_chains(chains,min_draws=4)
    chains=_scaled_chains(chains)
    m,n,_=chains.shape
    within=chains.var(1,unbiased=True).mean(0)
    between=chains.mean(1).var(0,unbiased=True) if m>1 else torch.zeros_like(within)
    variance=(n-1)/n*within+between
    acov=_autocovariance(chains).mean(0)
    rho=1-(within-acov)/variance
    rho[0]=1
    pair_count=n//2
    pairs=rho[:2*pair_count:2]+rho[1:2*pair_count:2]
    keep=(pairs>0).to(torch.int64).cumprod(0).bool()
    monotone=pairs.cummin(0).values
    tau=-1+2*torch.where(keep,monotone,torch.zeros_like(monotone)).sum(0)
    ess=(m*n)/tau.clamp_min(1)
    valid=(variance>0)&torch.isfinite(variance)&torch.isfinite(rho).all(0)&torch.isfinite(tau)&torch.isfinite(ess)
    return torch.where(valid,ess,torch.full_like(ess,float('nan')))


def _rank_normalize(draws):
    flat=draws.reshape(-1,draws.shape[-1])
    ranks=torch.empty_like(flat)
    for dimension in range(flat.shape[-1]):
        ordered,indices=flat[:,dimension].sort()
        _,inverse,counts=torch.unique_consecutive(ordered,return_inverse=True,return_counts=True)
        end=counts.cumsum(0).to(draws.dtype)
        average_rank=end-(counts.to(draws.dtype)-1)/2
        ranks[indices,dimension]=average_rank[inverse]
    probabilities=(ranks-.375)/(flat.shape[0]+.25)
    return (2**.5*torch.erfinv(2*probabilities-1)).reshape_as(draws)


def _rhat(draws):
    n=draws.shape[1]
    within=draws.var(1,unbiased=True).mean(0)
    variance=(n-1)/n*within+draws.mean(1).var(0,unbiased=True)
    return (variance/within).sqrt()


@torch.no_grad()
def rank_normalized_split_rhat(chains):
    """Maximum of rank-normalized split and folded-split Rhat, per dimension.

    Require >=2 independent chains and >=4 retained draws. Split into equal
    halves, discarding the central draw for odd lengths. Pooled tied values
    receive average ranks; Blom scores use (rank-3/8)/(count+1/4). Folding uses
    the pooled median of split draws. Undefined constant cases return NaN,
    including cases where the folded rank statistic has no variation.
    """
    _validate_chains(chains,min_chains=2,min_draws=4)
    half=chains.shape[1]//2
    split=torch.cat((chains[:,:half],chains[:,-half:]),dim=0)
    # Preserve raw ranks; shared scaling keeps folding subtraction finite even
    # for finite inputs near the dtype maximum.
    scaled=_scaled_chains(split)
    ordered=scaled.flatten(0,1).sort(dim=0).values
    lower=ordered[(len(ordered)-1)//2]
    upper=ordered[len(ordered)//2]
    # Ranking distance to the median interval is equivalent to ranking distance
    # to its midpoint: it removes the same half-gap from every observed value.
    # It also keeps the two central order statistics exactly tied.
    distance=torch.maximum(lower-scaled,scaled-upper).clamp_min(0)
    rank=_rhat(_rank_normalize(split))
    folded=_rhat(_rank_normalize(distance))
    return torch.maximum(rank,folded)


__all__=['autocorrelation','effective_sample_size','rank_normalized_split_rhat']


def _split(chains):
    half = chains.shape[1] // 2
    return torch.cat((chains[:, :half], chains[:, -half:]), dim=0)


def _split_ess(draws):
    """Geyer sequence with the final positive even lag (Vehtari et al., 2021).

    Mirrors the finite-sequence convention in ArviZ 0.22, except for the
    conservative N cap and undefined constant inputs. Work is vectorized over
    dimensions; each dimension stops its positive sequence independently.
    """
    draws = _scaled_chains(draws)
    m, n, d = draws.shape
    acov = _autocovariance(draws).mean(0)
    within = draws.var(1, unbiased=True).mean(0)
    variance = within * ((n - 1) / n) + draws.mean(1).var(0, unbiased=True)
    rho = 1 - (within - acov) / variance
    estimates = torch.zeros_like(rho)
    estimates[0] = 1
    estimates[1] = rho[1]
    even = torch.ones_like(within)
    odd = rho[1]
    active = even + odd > 0
    max_t = torch.full((d,), -1, dtype=torch.long, device=draws.device)
    for t in range(1, n - 3, 2):
        if not active.any():
            break
        even = torch.where(active, rho[t + 1], even)
        odd = torch.where(active, rho[t + 2], odd)
        accepted = active & (even + odd >= 0)
        estimates[t + 1] = torch.where(accepted, even, estimates[t + 1])
        estimates[t + 2] = torch.where(accepted, odd, estimates[t + 2])
        max_t = torch.where(active, t, max_t)
        active = active & (even + odd > 0)
    dimensions = torch.arange(d, device=draws.device)
    estimates[max_t + 1, dimensions] = torch.where(
        even > 0, even, estimates[max_t + 1, dimensions]
    )
    for t in range(1, int(max_t.max().item()) - 1, 2):
        previous = estimates[t - 1] + estimates[t]
        current = estimates[t + 1] + estimates[t + 2]
        reduce = (t <= max_t - 2) & (current > previous)
        estimates[t + 1] = torch.where(reduce, previous / 2, estimates[t + 1])
        estimates[t + 2] = torch.where(reduce, previous / 2, estimates[t + 2])
    lags = torch.arange(n, device=draws.device)[:, None]
    tau = -1 + 2 * torch.where(lags <= max_t, estimates, 0).sum(0)
    tau += estimates[max_t + 1, dimensions]
    ess = (m * n) / tau.clamp_min(1)
    valid = (variance > 0) & torch.isfinite(rho).all(0) & torch.isfinite(tau)
    return torch.where(valid, ess, torch.full_like(ess, float('nan')))


def _quality_input(chains):
    _validate_chains(chains, min_draws=10)
    # At least five draws per split: shorter ArviZ lag sequences ignore all
    # autocorrelation terms and can manufacture the maximum ESS.
    # Promote for ranks/FFT/probability arithmetic, preserving the public dtype.
    return chains.to(torch.float64)


def _probability(probability, *, tail=False):
    import numbers
    if (isinstance(probability, bool) or not isinstance(probability, numbers.Real)
            or not 0 < probability < (.5 if tail else 1)):
        raise ValueError('probability must be a real number in (0, .5)' if tail
                         else 'probability must be a real number in (0, 1)')
    return float(probability)


def _quantile_ess(chains, probability):
    # Threshold BEFORE splitting, including the central draw at odd lengths.
    scaled = _scaled_chains(chains)
    threshold = torch.quantile(scaled.flatten(0, 1), probability, dim=0)
    indicator = (scaled <= threshold).to(chains.dtype)
    return _split_ess(_split(indicator))


@torch.no_grad()
def bulk_effective_sample_size(chains):
    """Split rank-normalized bulk ESS, one value per dimension.

    Require float32/float64 (independent chains, >=10 draws, dimensions), with
    >=1 chain. Odd lengths discard the central draw. Average tied ranks and
    Blom normal scores follow Vehtari et al. (2021), doi:10.1214/20-BA1221.
    Geyer's positive/monotone sequence includes the final positive even lag.
    ESS is conservatively capped at the retained split count 2*M*floor(N/2),
    including antithetic chains. Constant dimensions give NaN. Ten draws gives at least five draws per split so the finite lag estimator
    can include autocorrelation; it is only a computational minimum, and short
    runs cannot establish convergence.
    """
    draws = _quality_input(chains)
    return _split_ess(_rank_normalize(_split(draws))).to(chains.dtype)


@torch.no_grad()
def tail_effective_sample_size(chains, *, probability=.05):
    """Minimum lower/upper quantile-indicator split ESS (Vehtari et al. 2021).

    probability must lie in (0, .5); thresholds are pooled linearly interpolated
    quantiles at p and 1-p, using indicators x <= threshold. Count, shape and
    cap match bulk_effective_sample_size. Constant indicators (including tied
    extreme quantiles) return NaN rather than manufactured convergence.
    """
    p = _probability(probability, tail=True)
    draws = _quality_input(chains)
    return torch.minimum(_quantile_ess(draws, p), _quantile_ess(draws, 1 - p)).to(chains.dtype)


@torch.no_grad()
def mcse_mean(chains):
    """Estimated mean Monte Carlo SE: pooled sample SD / sqrt(raw split ESS).

    Use >=1 independent chain and >=10 draws; rank ESS is inappropriate for this
    estimand. SD uses all pooled draws (ddof=1), ESS uses equal split halves and
    the conservative retained-count cap. Constant dimensions give NaN. Finite
    variance and stationary mixing are assumptions: heavy tails can invalidate
    a mean SE even when this finite-run numerical estimate is finite.
    """
    draws = _quality_input(chains)
    scale = draws.abs().amax(dim=(0, 1))
    scaled = _scaled_chains(draws)
    result = scaled.flatten(0, 1).std(0, unbiased=True) / _split_ess(_split(scaled)).sqrt()
    return (result * scale).to(chains.dtype)


def _beta_cdf(x, a, b):
    """Regularized incomplete beta via its continued fraction (DLMF 8.17.22)."""
    import math
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    if x > (a + 1) / (a + b + 2):
        return 1 - _beta_cdf(1 - x, b, a)
    tiny = 1e-300
    c = 1.
    denominator = 1 - (a + b) * x / (a + 1)
    denominator = denominator if abs(denominator) > tiny else tiny
    inverse = 1 / denominator
    fraction = inverse
    for m in range(1, 1001):
        for coefficient in (m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)),
                            -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))):
            denominator = 1 + coefficient * inverse
            denominator = denominator if abs(denominator) > tiny else tiny
            c = 1 + coefficient / c
            c = c if abs(c) > tiny else tiny
            inverse = 1 / denominator
            delta = inverse * c
            fraction *= delta
        if abs(delta - 1) < 3e-14:
            log_factor = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            log_factor += a * math.log(x) + b * math.log1p(-x)
            return math.exp(log_factor) * fraction / a
    raise RuntimeError('incomplete beta did not converge; quantile MCSE is undefined')


def _beta_inverse(probability, a, b):
    low, high = 0., 1.
    for _ in range(64):
        midpoint = (low + high) / 2
        if _beta_cdf(midpoint, a, b) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


@torch.no_grad()
def mcse_quantile(chains, probability=.5):
    """Quantile Monte Carlo SE from an ESS-adjusted beta order-statistic interval.

    Following Vehtari et al. (2021), section 4.4, and ArviZ 0.22's quantile
    MCSE: compute split ESS E of x <= pooled quantile(p); find beta(E*p+1,
    E*(1-p)+1) probability bounds .1586553 and .8413447; map bounds to pooled
    sorted draws at floor(N*lower-1), ceil(N*upper-1), clipped to [0,N-1].
    Return half the interval width. This accounts for indicator autocorrelation,
    and needs no density estimate or independent-quantile assumption. ESS is
    capped at the retained split count. Require >=10 draws, >=1 independent
    chain, 0<p<1; constant indicators give NaN. Ties can give zero width despite
    positive indicator ESS; discrete quantiles need careful interpretation.
    Arithmetic preserves extreme finite scales; results keep dtype/device.
    Beta inversion uses scalar CPU arithmetic, so this diagnostic synchronizes
    accelerator inputs and is intended for reporting rather than sampler steps.
    """
    p = _probability(probability)
    draws = _quality_input(chains)
    ess = _quantile_ess(draws, p)
    ordered = draws.flatten(0, 1).sort(dim=0).values
    count, dimensions = ordered.shape
    result = draws.new_full((dimensions,), float('nan'))
    for d, size in enumerate(ess.cpu().tolist()):
        if not (size > 0):
            continue
        import math
        lower = _beta_inverse(.1586553, size*p + 1, size*(1-p) + 1)
        upper = _beta_inverse(.8413447, size*p + 1, size*(1-p) + 1)
        lo = max(0, math.floor(count * lower - 1))
        hi = min(count - 1, math.ceil(count * upper - 1))
        result[d] = ordered[hi, d] / 2 - ordered[lo, d] / 2
    return result.to(chains.dtype)


__all__ += ['bulk_effective_sample_size', 'tail_effective_sample_size', 'mcse_mean', 'mcse_quantile']
