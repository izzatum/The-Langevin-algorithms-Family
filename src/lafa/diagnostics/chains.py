"""Diagnostics for INDEPENDENT stationary chains, never interacting particles.

Inputs are (chains, draws, dimensions), with adaptation and warmup removed.
Rhat follows Vehtari et al. (2021), Bayesian Analysis 16(2), rank normalization,
folding, and splitting. ESS uses a pooled variance estimate and Geyer's initial
positive, monotone autocorrelation pairs. It is raw-parameter mean ESS, not
rank/bulk/tail ESS, and is conservatively capped at the total draw count.
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
    median=torch.quantile(split.reshape(-1,split.shape[-1]),.5,dim=0)
    rank=_rhat(_rank_normalize(split))
    folded=_rhat(_rank_normalize((split-median).abs()))
    return torch.maximum(rank,folded)


__all__=['autocorrelation','effective_sample_size','rank_normalized_split_rhat']
