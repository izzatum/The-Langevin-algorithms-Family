"""Linear Gaussian state-space model with an independent Kalman reference."""
from dataclasses import dataclass
import math
import torch


def _real_dense(value,device,dtype):
    # Inspect before casting so complex values are never silently truncated.
    original=torch.as_tensor(value)
    if original.layout!=torch.strided or original.is_complex() or original.device.type=='meta':
        raise ValueError('state-space inputs must be dense, materialized real tensors')
    return torch.as_tensor(value,device=device,dtype=dtype)


@dataclass(frozen=True)
class KalmanResult:
    means: torch.Tensor
    covariances: torch.Tensor
    log_likelihood: torch.Tensor
    predicted_means: torch.Tensor
    predicted_covariances: torch.Tensor


class LinearGaussianStateSpace:
    """x0~N(m0,P0); xt=F*x(t-1)+q; yt=H*xt+r.

    First observation is at t=0. All noises are zero mean, mutually independent.
    Default model is scalar; noise covariances must be strictly positive definite.
    """
    def __init__(self,transition=None,observation=None,process_covariance=None,observation_covariance=None,initial_mean=None,initial_covariance=None,*,device='cpu',dtype=torch.float64):
        if dtype not in (torch.float32,torch.float64):
            raise ValueError('state-space dtype must be float32 or float64')
        self.device,self.dtype=torch.device(device),dtype
        def t(x): return _real_dense(x,device,dtype).detach().clone()
        self.transition=t([[.85]] if transition is None else transition)
        if self.transition.ndim!=2 or not self.transition.shape[0] or self.transition.shape[0]!=self.transition.shape[1] or not torch.isfinite(self.transition).all():
            raise ValueError('transition must be a finite nonempty square matrix')
        self.dimension=len(self.transition)
        d=self.dimension
        self.observation=t(torch.eye(d) if observation is None else observation)
        if self.observation.ndim!=2 or not self.observation.shape[0] or self.observation.shape[1]!=d or not torch.isfinite(self.observation).all():
            raise ValueError('observation must be finite with positive observation and compatible state dimensions')
        o=self.observation.shape[0]
        self.observation_dimension=o
        self.process_covariance=t(.1*torch.eye(d) if process_covariance is None else process_covariance)
        self.observation_covariance=t(.2*torch.eye(o) if observation_covariance is None else observation_covariance)
        self.initial_mean=t(torch.zeros(d) if initial_mean is None else initial_mean)
        self.initial_covariance=t(torch.eye(d) if initial_covariance is None else initial_covariance)
        if self.initial_mean.shape!=(d,) or not torch.isfinite(self.initial_mean).all():
            raise ValueError('initial_mean must be finite with shape (state_dimension,)')
        for cov,n in [(self.process_covariance,d),(self.observation_covariance,o),(self.initial_covariance,d)]:
            if cov.shape!=(n,n) or not torch.allclose(cov,cov.T) or not torch.isfinite(cov).all():
                raise ValueError('noise/initial covariances must be finite symmetric matrices')
            torch.linalg.cholesky(cov)
        self.F,self.H,self.Q,self.R=self.transition,self.observation,self.process_covariance,self.observation_covariance

    def simulate(self,steps,*,seed=0,generator=None):
        if not isinstance(steps,int) or steps<1: raise ValueError('steps must be a positive integer')
        generator=torch.Generator().manual_seed(seed) if generator is None else generator
        # Seeded fixtures use CPU randomness before transfer, keeping data identical across devices.
        gd=generator.device
        def noise(n): return torch.randn(n,generator=generator,device=gd,dtype=self.dtype).to(self.device)
        q=torch.linalg.cholesky(self.Q); r=torch.linalg.cholesky(self.R)
        x=self.initial_mean+torch.linalg.cholesky(self.initial_covariance)@noise(self.dimension)
        states,observations=[],[]
        for i in range(steps):
            if i: x=self.F@x+q@noise(self.dimension)
            states.append(x)
            observations.append(self.H@x+r@noise(self.observation_dimension))
        return torch.stack(states),torch.stack(observations)

    def kalman_filter(self,observations):
        observations=_real_dense(observations,self.device,self.dtype)
        if observations.ndim!=2 or observations.shape[1]!=self.observation_dimension or not len(observations) or not torch.isfinite(observations).all():
            raise ValueError('observations must be finite with shape (time,observation_dimension)')
        mean,cov=self.initial_mean,self.initial_covariance
        means,covs,pm,pc=[],[],[],[]
        log_likelihood=mean.new_zeros(())
        eye=torch.eye(self.dimension,device=self.device,dtype=self.dtype)
        for i,y in enumerate(observations):
            if i: mean=self.F@mean; cov=self.F@cov@self.F.T+self.Q
            pm.append(mean); pc.append(cov)
            residual=y-self.H@mean
            innovation=self.H@cov@self.H.T+self.R
            chol=torch.linalg.cholesky(innovation)
            gain=torch.cholesky_solve(self.H@cov,chol).T
            mean=mean+gain@residual
            correction=eye-gain@self.H
            cov=correction@cov@correction.T+gain@self.R@gain.T # Joseph form
            cov=(cov+cov.T)/2
            white=torch.linalg.solve_triangular(chol,residual[:,None],upper=False)
            log_likelihood=log_likelihood-.5*(self.observation_dimension*math.log(2*math.pi)+2*chol.diag().log().sum()+white.square().sum())
            means.append(mean); covs.append(cov)
        return KalmanResult(torch.stack(means),torch.stack(covs),log_likelihood,torch.stack(pm),torch.stack(pc))
