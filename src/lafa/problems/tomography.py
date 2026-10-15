"""Straight-ray spatial inverse fixtures with actual ray/cell intersection lengths.

Unit square, row-major (y,x) cells. Sparse forward evaluation supports arbitrary
leading batch axes and autograd on CPU/CUDA. Geometry is assembled once on CPU.
See docs/references.md R01 for the linear traveltime motivation; the nonlinear
transmission fixture is a separate synthetic benchmark.
"""
import math
import torch
from .distributions import Gaussian


def _grid_shape(shape):
    shape=tuple(shape)
    if len(shape)!=2 or any(not isinstance(n,int) or n<1 for n in shape):
        raise ValueError('grid_shape must be two positive integers')
    return shape


def ray_cell_operator(grid_shape,rays,*,device=None,dtype=None):
    """COO path lengths for segments `rays[n, endpoint, (x,y)]` in [0,1]².

    Splits every segment at all grid boundaries, assigns each open subsegment
    by its midpoint, and retains its Euclidean length inside the domain. Rays
    on internal grid lines are consistently assigned to their upper/right cell.
    """
    ny,nx=_grid_shape(grid_shape)
    rays=torch.as_tensor(rays)
    device=rays.device if device is None else torch.device(device)
    dtype=rays.dtype if dtype is None and rays.is_floating_point() else (torch.float64 if dtype is None else dtype)
    if rays.ndim!=3 or rays.shape[1:]!=(2,2) or not torch.isfinite(rays).all():
        raise ValueError('rays must have finite shape (n,2,2)')
    rows,cols,values=[],[],[]
    for k,(start,end) in enumerate(rays.detach().cpu().double().tolist()):
        dx,dy=end[0]-start[0],end[1]-start[1]
        length=math.hypot(dx,dy)
        if length==0: continue
        ts=[0.,1.]
        for origin,delta,n in [(start[0],dx,nx),(start[1],dy,ny)]:
            if delta!=0:
                ts.extend(t for j in range(n+1) if 0<(t:=(j/n-origin)/delta)<1)
        ts=sorted(set(round(t,14) for t in ts))
        for ta,tb in zip(ts[:-1],ts[1:]):
            t=(ta+tb)/2
            xx,yy=start[0]+t*dx,start[1]+t*dy
            if 0<=xx<=1 and 0<=yy<=1 and tb>ta:
                ix,iy=min(nx-1,int(xx*nx)),min(ny-1,int(yy*ny))
                rows.append(k); cols.append(iy*nx+ix); values.append((tb-ta)*length)
    indices=torch.tensor([rows,cols],device=device,dtype=torch.int64)
    return torch.sparse_coo_tensor(indices,torch.tensor(values,device=device,dtype=dtype),(len(rays),nx*ny),device=device,dtype=dtype).coalesce()


def boundary_rays(grid_shape):
    """All bottom/top and left/right detector pairs at cell-center offsets."""
    ny,nx=_grid_shape(grid_shape)
    n=max(ny,nx)
    u=(torch.arange(n,dtype=torch.float64)+.5)/n
    rays=[]
    for a in u.tolist():
        for b in u.tolist():
            rays.append([[a,0.],[b,1.]])
            rays.append([[0.,a],[1.,b]])
    return torch.tensor(rays,dtype=torch.float64)


class _Spatial:
    def _geometry(self,grid_shape,rays,device,dtype):
        self.grid_shape=_grid_shape(grid_shape)
        self.dimension=math.prod(self.grid_shape)
        self.device,self.dtype=torch.device(device),dtype
        self.rays=(boundary_rays(self.grid_shape) if rays is None else torch.as_tensor(rays)).to(device=device,dtype=dtype)
        self.operator=ray_cell_operator(self.grid_shape,self.rays,device=device,dtype=dtype)
        self.ray_lengths=torch.sparse.sum(self.operator,dim=1).to_dense()
        ny,nx=self.grid_shape
        yy,xx=torch.meshgrid((torch.arange(ny,device=device,dtype=dtype)+.5)/ny,(torch.arange(nx,device=device,dtype=dtype)+.5)/nx,indexing='ij')
        b1=torch.exp(-((xx-.32).square()+(yy-.38).square())/(2*.18**2))
        b2=torch.exp(-((xx-.7).square()+(yy-.66).square())/(2*.16**2))
        self.basis=torch.stack((b1.reshape(-1),b2.reshape(-1)),dim=-1)

    def line_integral(self,x):
        if x.shape[-1]!=self.operator.shape[1]: raise ValueError('last axis must match the number of grid cells')
        return torch.sparse.mm(self.operator,x.reshape(-1,x.shape[-1]).T).T.reshape(*x.shape[:-1],self.operator.shape[0])

    def field(self,x):
        """Plot-ready spatial field, with batch axes preserved."""
        return x.reshape(*x.shape[:-1],*self.grid_shape)


class TraveltimeTomography2D(_Spatial):
    """y=A*s+noise, Gaussian prior on slowness s (a linear reference model).

    The Gaussian prior is unconstrained and can assign negative slowness. For a
    positive physical field use NonlinearTomography2D instead. No dense posterior
    covariance is allocated until posterior() is explicitly called.
    """
    def __init__(self,grid_shape=(8,8),*,rays=None,noise_scale=.05,prior_scale=.3,prior_mean=1.,seed=0,device='cpu',dtype=torch.float64):
        if not math.isfinite(noise_scale) or not math.isfinite(prior_scale) or noise_scale<=0 or prior_scale<=0: raise ValueError('noise/prior scales must be finite and positive')
        if not math.isfinite(prior_mean): raise ValueError('prior_mean must be finite')
        self._geometry(grid_shape,rays,device,dtype)
        self.noise_scale,self.prior_scale=float(noise_scale),float(prior_scale)
        self.prior_mean=torch.full((self.dimension,),float(prior_mean),device=device,dtype=dtype)
        self.truth=self.prior_mean+self.basis@torch.tensor([.3,-.2],device=device,dtype=dtype)
        self.seed=seed
        generator=torch.Generator().manual_seed(seed)
        self.observations=self.forward(self.truth)+noise_scale*torch.randn(len(self.rays),dtype=dtype,generator=generator).to(device)

    def forward(self,x): return self.line_integral(x)

    def potential(self,x):
        return .5*((self.forward(x)-self.observations)/self.noise_scale).square().sum(-1)+.5*((x-self.prior_mean)/self.prior_scale).square().sum(-1)

    __call__=potential

    def posterior(self,*,max_dimension=1024):
        """Opt-in exact dense posterior; guarded against accidental large allocation."""
        if self.dimension>max_dimension: raise ValueError('dense posterior exceeds max_dimension; explicitly raise the limit if intended')
        a=self.operator.to_dense()
        identity=torch.eye(self.dimension,device=self.device,dtype=self.dtype)
        precision=a.T@a/self.noise_scale**2+identity/self.prior_scale**2
        rhs=a.T@self.observations/self.noise_scale**2+self.prior_mean/self.prior_scale**2
        chol=torch.linalg.cholesky(precision)
        mean=torch.cholesky_solve(rhs[:,None],chol).squeeze(-1)
        cov=torch.cholesky_solve(identity,chol)
        return Gaussian(mean,cov,device=self.device,dtype=self.dtype)


class NonlinearTomography2D(_Spatial):
    """Transmission y=exp(-A exp(x))+Gaussian noise, latent x~N(offset,s²I).

    x is the log attenuation field: the prior is defined on x, so there is no
    change-of-variables Jacobian in this potential. Observations can leave [0,1]
    because their noise is Gaussian. Truth has two smooth spatial inclusions.
    """
    def __init__(self,grid_shape=(8,8),*,rays=None,noise_scale=.04,prior_scale=.7,offset=-.6,seed=0,device='cpu',dtype=torch.float64):
        if not math.isfinite(noise_scale) or not math.isfinite(prior_scale) or noise_scale<=0 or prior_scale<=0: raise ValueError('noise/prior scales must be finite and positive')
        if not math.isfinite(offset): raise ValueError('offset must be finite')
        self._geometry(grid_shape,rays,device,dtype)
        self.noise_scale,self.prior_scale,self.offset=float(noise_scale),float(prior_scale),float(offset)
        self.truth_coefficients=torch.tensor([.7,-.5],device=device,dtype=dtype)
        self.truth=self.offset+self.basis@self.truth_coefficients
        self.seed=seed
        generator=torch.Generator().manual_seed(seed)
        self.observations=self.forward(self.truth)+noise_scale*torch.randn(len(self.rays),dtype=dtype,generator=generator).to(device)

    def forward(self,x): return torch.exp(-self.line_integral(x.exp()))

    def potential(self,x):
        return .5*((self.forward(x)-self.observations)/self.noise_scale).square().sum(-1)+.5*((x-self.offset)/self.prior_scale).square().sum(-1)

    __call__=potential

    def field(self,x): return super().field(x.exp())

    def reduced(self,*,prior_scale=None):
        """Two-coefficient Gaussian-prior model using the same full spatial map/data.

        Its coefficient prior is specified explicitly; it is a separate reduced
        model, not a marginal of the independent full-grid Gaussian prior.
        """
        return ReducedNonlinearTomography2D(self,prior_scale=self.prior_scale if prior_scale is None else prior_scale)


class ReducedNonlinearTomography2D:
    dimension=2
    def __init__(self,spatial,*,prior_scale=.7):
        if not math.isfinite(prior_scale) or prior_scale<=0: raise ValueError('prior_scale must be finite and positive')
        self.spatial=spatial
        self.device,self.dtype=spatial.device,spatial.dtype
        self.grid_shape,self.operator,self.rays=spatial.grid_shape,spatial.operator,spatial.rays
        self.observations,self.noise_scale=spatial.observations,spatial.noise_scale
        self.prior_scale=float(prior_scale)
        self.truth=spatial.truth_coefficients.clone()
        self.basis,self.offset=spatial.basis,spatial.offset

    def latent_field(self,coefficients): return self.offset+coefficients@self.basis.T
    def field(self,coefficients): return self.spatial.field(self.latent_field(coefficients))
    def forward(self,coefficients): return self.spatial.forward(self.latent_field(coefficients))

    def potential(self,coefficients):
        return .5*((self.forward(coefficients)-self.observations)/self.noise_scale).square().sum(-1)+.5*(coefficients/self.prior_scale).square().sum(-1)

    __call__=potential
