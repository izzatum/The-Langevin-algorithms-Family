"""Independent midpoint integration references for two-dimensional targets."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class QuadratureResult:
    mean: torch.Tensor
    covariance: torch.Tensor
    log_normalizer: torch.Tensor
    boundary_mass: torch.Tensor
    moments: dict
    bounds: tuple
    resolution: tuple


def quadrature_2d(potential,bounds,resolution=200,*,chunk_size=4096,device=None,dtype=None):
    """Midpoint rule with logsumexp and cell area; conditional on given bounds.

    `boundary_mass` is probability in the outermost grid cells, useful alongside
    expanded bounds and resolution doubling; it is not a bound on omitted mass.
    `moments` contains coordinatewise raw second/third/fourth moments. Evaluates
    in chunks without autograd graphs, including spatial forward maps.
    """
    owner=getattr(potential,'__self__',potential)
    device=getattr(owner,'device','cpu') if device is None else device
    dtype=getattr(owner,'dtype',torch.float64) if dtype is None else dtype
    if len(bounds)!=2 or any(len(b)!=2 or not all(math.isfinite(v) for v in b) or b[1]<=b[0] for b in bounds):
        raise ValueError('bounds must contain two finite increasing intervals')
    shape=(resolution,resolution) if isinstance(resolution,int) else tuple(resolution)
    if len(shape)!=2 or any(not isinstance(n,int) or n<2 for n in shape) or chunk_size<1:
        raise ValueError('resolution must be >=2 per axis and chunk_size positive')
    axes=[]
    spacing=[]
    for (lo,hi),n in zip(bounds,shape):
        h=(hi-lo)/n
        axes.append(lo+(torch.arange(n,device=device,dtype=dtype)+.5)*h)
        spacing.append(h)
    xy=torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=-1).reshape(-1,2)
    with torch.no_grad():
        values=torch.cat([potential(c) for c in xy.split(chunk_size)])
        if values.shape!=(len(xy),) or torch.isnan(values).any() or torch.isneginf(values).any():
            raise ValueError('potential must return one non-NaN value per point, without -inf')
        logtotal=torch.logsumexp(-values,0)
        if not torch.isfinite(logtotal): raise ValueError('quadrature contains no finite positive density')
        w=(-values-logtotal).exp()
        mean=(w[:,None]*xy).sum(0)
        delta=xy-mean
        covariance=torch.einsum('n,ni,nj->ij',w,delta,delta)
        border=torch.zeros(shape,device=device,dtype=torch.bool)
        border[0,:]=True; border[-1,:]=True; border[:,0]=True; border[:,-1]=True
        return QuadratureResult(mean,covariance,logtotal+math.log(math.prod(spacing)),w[border.reshape(-1)].sum(),
            {name:(w[:,None]*xy.pow(power)).sum(0) for name,power in [('second',2),('third',3),('fourth',4)]},tuple(map(tuple,bounds)),shape)
