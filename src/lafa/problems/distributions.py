"""Analytic fixtures. See docs/references.md R01 and paper-interpretation.md."""
import math
import torch


def _tensor(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype).detach().clone()


def _sample_shape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


class Gaussian:
    """Gaussian with U(x)=.5 ||L^-1(x-mean)||²; no dense inverse in evaluation."""
    def __init__(self, mean, covariance, *, device='cpu', dtype=torch.float64):
        self.mean = _tensor(mean,device,dtype)
        self.covariance = _tensor(covariance,device,dtype)
        self.device,self.dtype=self.mean.device,dtype
        self.dimension=self.mean.numel()
        if self.mean.ndim!=1 or not self.dimension or self.covariance.shape!=(self.dimension,self.dimension):
            raise ValueError('mean and covariance must have shapes (D,) and (D,D)')
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.covariance).all():
            raise ValueError('Gaussian parameters must be finite')
        if not torch.allclose(self.covariance,self.covariance.T):
            raise ValueError('covariance must be symmetric positive definite')
        self.cholesky=torch.linalg.cholesky(self.covariance)
        self.log_normalizer=.5*self.dimension*math.log(2*math.pi)+self.cholesky.diag().log().sum()

    def potential(self,x):
        delta=x-self.mean
        white=torch.linalg.solve_triangular(self.cholesky,delta.reshape(-1,self.dimension).T,upper=False).T
        return .5*white.square().sum(-1).reshape(x.shape[:-1])

    __call__=potential

    def score(self,x):
        delta=x-self.mean
        return -torch.cholesky_solve(delta.reshape(-1,self.dimension).T,self.cholesky).T.reshape(x.shape)

    def sample(self,shape=(),*,generator=None):
        z=torch.randn((*_sample_shape(shape),self.dimension),generator=generator,device=self.device,dtype=self.dtype)
        return self.mean+z@self.cholesky.T


def paper_gaussian(*,device='cpu',dtype=torch.float64):
    """R01 Section 4 Gaussian, using its A, data and small regularizer L."""
    a=torch.tensor([[2.,.5],[.5,2.]],device=device,dtype=dtype)
    l=1e-3*torch.tensor([[.5,0.],[2.,0.]],device=device,dtype=dtype)
    precision=a.T@a+l.T@l
    chol=torch.linalg.cholesky(precision)
    mean=torch.cholesky_solve((a.T@a.new_ones(2)).unsqueeze(-1),chol).squeeze(-1)
    covariance=torch.cholesky_solve(torch.eye(2,device=device,dtype=dtype),chol)
    return Gaussian(mean,covariance,device=device,dtype=dtype)


class GaussianMixture2D:
    """Four corner modes; normalized component densities and exact total moments."""
    dimension=2
    def __init__(self,means=None,weights=None,covariance=None,*,device='cpu',dtype=torch.float64):
        self.device,self.dtype=torch.device(device),dtype
        self.means=_tensor([[-3.,-3.],[-3.,3.],[3.,-3.],[3.,3.]] if means is None else means,device,dtype)
        if self.means.ndim!=2 or self.means.shape[1]!=2 or not self.means.shape[0] or not torch.isfinite(self.means).all():
            raise ValueError('means must have finite shape (components,2)')
        count=len(self.means)
        self.weights=_tensor([1.]*count if weights is None else weights,device,dtype)
        if self.weights.shape!=(count,) or not torch.isfinite(self.weights).all() or not (self.weights>0).all():
            raise ValueError('weights must be finite and strictly positive')
        self.weights=self.weights/self.weights.max()
        self.weights=self.weights/self.weights.sum()
        base=Gaussian([0.,0.],[[.5,0.],[0.,.5]] if covariance is None else covariance,device=device,dtype=dtype)
        self.component_covariance,self.cholesky=base.covariance,base.cholesky
        self._component_log_normalizer=base.log_normalizer
        self.mean=(self.weights[:,None]*self.means).sum(0)
        centered=self.means-self.mean
        self.covariance=base.covariance+torch.einsum('k,ki,kj->ij',self.weights,centered,centered)
        self.log_normalizer=0.

    def potential(self,x):
        delta=x[...,None,:]-self.means
        white=torch.linalg.solve_triangular(self.cholesky,delta.reshape(-1,2).T,upper=False).T
        log_density=-.5*white.square().sum(-1).reshape(delta.shape[:-1])-self._component_log_normalizer
        return -torch.logsumexp(self.weights.log()+log_density,dim=-1)

    __call__=potential

    def sample(self,shape=(),*,generator=None):
        shape=_sample_shape(shape)
        indices=torch.multinomial(self.weights,math.prod(shape) if shape else 1,replacement=True,generator=generator).reshape(shape)
        z=torch.randn((*shape,2),device=self.device,dtype=self.dtype,generator=generator)
        return self.means[indices]+z@self.cholesky.T


class Rosenbrock2D:
    """Corrected R01 Eq. (10): U=alpha*(x1²-x2)²+(x1-beta)^4.

    x1-beta is quartic-exponential, NOT Gaussian. Reference moments and normalizer
    follow the unit-Jacobian change (z,e)=(x1-beta,x2-x1²).
    """
    dimension=2
    def __init__(self,alpha=10.,beta=.25,*,device='cpu',dtype=torch.float64):
        if not math.isfinite(alpha) or alpha<=0 or not math.isfinite(beta):
            raise ValueError('alpha must be positive and beta finite')
        self.alpha,self.beta=float(alpha),float(beta)
        self.device,self.dtype=torch.device(device),dtype
        c=math.gamma(.75)/math.gamma(.25)
        self.mean=torch.tensor([beta,beta*beta+c],device=device,dtype=dtype)
        self.covariance=torch.tensor([[c,2*beta*c],[2*beta*c,.25-c*c+4*beta*beta*c+1/(2*alpha)]],device=device,dtype=dtype)
        self.log_normalizer=math.log(math.gamma(.25)/2)+.5*math.log(math.pi/alpha)

    def potential(self,x):
        return self.alpha*(x[...,0].square()-x[...,1]).square()+(x[...,0]-self.beta).pow(4)

    __call__=potential

    def reference(self,*,resolution=400,bounds=((-2.5,3.),(-1.5,9.))):
        from .quadrature import quadrature_2d
        return quadrature_2d(self,bounds,resolution=resolution)


class Banana2D:
    """Separate Gaussian banana: x=(z1, z2+b*(z1²-1)), z~N(0,I)."""
    dimension=2
    def __init__(self,bend=1.,*,device='cpu',dtype=torch.float64):
        if not math.isfinite(bend): raise ValueError('bend must be finite')
        self.bend=bend
        self.device,self.dtype=torch.device(device),dtype
        self.mean=torch.zeros(2,device=device,dtype=dtype)
        self.covariance=torch.diag(torch.tensor([1.,1+2*bend*bend],device=device,dtype=dtype))
        self.log_normalizer=math.log(2*math.pi)

    def potential(self,x):
        return .5*(x[...,0].square()+(x[...,1]-self.bend*(x[...,0].square()-1)).square())

    __call__=potential

    def sample(self,shape=(),*,generator=None):
        z=torch.randn((*_sample_shape(shape),2),device=self.device,dtype=self.dtype,generator=generator)
        return torch.stack((z[...,0],z[...,1]+self.bend*(z[...,0].square()-1)),dim=-1)
