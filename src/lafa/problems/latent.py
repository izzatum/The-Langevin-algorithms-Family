"""Small nonsmooth latent learning fixture for proximal particle methods."""
import math
import torch


class LatentGaussianLaplace:
    """y_i | theta,z_i ~ N(theta+z_i, noise_scale²), penalty*sum|z_i|.

    Shared theta is learned; latent coordinates z are sampled/regularized.
    `smooth_potential` excludes the Laplace penalty and includes optional
    theta~N(0,theta_scale²) regularization (theta_scale=inf gives no penalty).
    This is a learning fixture, not a claim that PIPLA samples theta's posterior.
    """
    def __init__(self,observations=None,*,noise_scale=1.,penalty=1.,theta_scale=math.inf,seed=0,device='cpu',dtype=torch.float64):
        if not math.isfinite(noise_scale) or not math.isfinite(penalty) or noise_scale<=0 or penalty<=0 or math.isnan(theta_scale) or theta_scale<=0:
            raise ValueError('noise, penalty and theta scales must be positive')
        self.device,self.dtype=torch.device(device),dtype
        if observations is None:
            g=torch.Generator().manual_seed(seed)
            observations=.6+torch.randn(4,generator=g,dtype=dtype)
        self.observations=torch.as_tensor(observations,device=device,dtype=dtype).detach().clone()
        if self.observations.ndim!=1 or not self.observations.numel() or not torch.isfinite(self.observations).all():
            raise ValueError('observations must be a finite vector')
        self.dimension=self.latent_dimension=self.observations.numel()
        self.noise_scale,self.penalty,self.theta_scale=float(noise_scale),float(penalty),float(theta_scale)

    def smooth_potential(self,theta,latent):
        theta=torch.as_tensor(theta,device=self.device,dtype=self.dtype)
        return .5*((latent+theta-self.observations)/self.noise_scale).square().sum(-1)+.5*theta.square().sum(-1)/self.theta_scale**2

    def potential(self,theta,latent):
        return self.smooth_potential(theta,latent)+self.penalty*latent.abs().sum(-1)

    def gradient_theta(self,theta,latent):
        return ((latent+theta-self.observations)/self.noise_scale**2).sum(-1,keepdim=True)+theta/self.theta_scale**2

    def gradient_latent(self,theta,latent):
        return (latent+theta-self.observations)/self.noise_scale**2

    def prox_latent(self,latent,step_size):
        return latent.sign()*(latent.abs()-step_size*self.penalty).clamp_min(0)

    def prox_joint(self,theta,latent,step_size):
        """Exact prox of the nonsmooth Laplace term, per particle.

        Returns theta with shape (...,1), matching latent leading particle axes;
        the shared scalar parameter is unchanged by this nonsmooth-only prox.
        Use full_joint_prox for PIPULA's full-potential proximal step.
        """
        theta=self._scalar_theta(theta,latent)
        return theta.clone(),self.prox_latent(latent,step_size)

    def _scalar_theta(self,theta,latent):
        theta=torch.as_tensor(theta,device=self.device,dtype=self.dtype)
        if theta.ndim==0: theta=theta.reshape(1)
        if theta.shape[-1]!=1 or latent.shape[-1]!=self.dimension:
            raise ValueError('theta must be scalar (...,1) and latent must end in observation dimension')
        return torch.broadcast_to(theta,(*latent.shape[:-1],1))

    def full_joint_prox(self,theta,latent,step_size):
        """Exact full Gaussian-plus-Laplace joint prox for each scalar theta.

        Minimizes U(t,z)+((t-theta)^2+||z-latent||²)/(2h), including
        theta_scale regularization. Returns (...,1), (...,D).

        For fixed t, z=soft(v*b+h*(y-t), h*v*penalty)/(v+h), v=sigma².
        The reduced derivative F(t) is continuous piecewise affine, with slope
        at least 1/h+1/theta_scale²+D/(v+h)>0, hence a unique root.
        Its 2D breakpoints are y+v*b/h +/- v*penalty. Starting at t=-infinity,
        F has known affine coefficients; each ordered breakpoint changes the
        slope by +/- h/(v*(v+h)), preserving continuity. Locate the interval
        from the sign of F at each breakpoint and solve its affine root.
        No iteration count, tolerance truncation, or approximate-prox claim is
        involved; errors are solely floating-point arithmetic.
        """
        h=float(step_size)
        if not math.isfinite(h) or h<=0: raise ValueError('step_size must be finite and positive')
        a=self._scalar_theta(theta,latent)
        v=self.noise_scale**2
        lam=self.penalty
        center=self.observations+(v/h)*latent
        lower,upper=center-v*lam,center+v*lam
        breaks,order=torch.cat((lower,upper),dim=-1).sort(dim=-1)
        jump=h/(v*(v+h))
        jumps=torch.cat((torch.full_like(latent,jump),torch.full_like(latent,-jump)),dim=-1).gather(-1,order)
        base_slope=1/h+1/self.theta_scale**2+self.dimension/(v+h)
        base_intercept=-a/h+((latent-self.observations-h*lam)/(v+h)).sum(-1,keepdim=True)
        zeros=torch.zeros_like(a)
        slopes=base_slope+torch.cat((zeros,jumps.cumsum(-1)),dim=-1)
        intercepts=base_intercept-torch.cat((zeros,(jumps*breaks).cumsum(-1)),dim=-1)
        at_breaks=slopes[...,:-1]*breaks+intercepts[...,:-1]
        interval=(at_breaks<0).sum(-1,keepdim=True)
        t=-intercepts.gather(-1,interval)/slopes.gather(-1,interval)
        raw=v*latent+h*(self.observations-t)
        z=raw.sign()*(raw.abs()-h*v*lam).clamp_min(0)/(v+h)
        return t,z

    def latent_map(self,theta):
        """Exact conditional MAP from soft thresholding y-theta."""
        return self.prox_latent(self.observations-theta,self.noise_scale**2)

    def log_marginal_likelihood(self,theta):
        """Exact normalized Normal/Laplace convolution, integrating out all z.

        This independent reference excludes any theta regularizer. Stable normal
        log-CDFs retain differentiability even for extreme residuals.
        """
        theta=torch.as_tensor(theta,device=self.device,dtype=self.dtype)
        residual=self.observations-theta
        lam,sigma=self.penalty,self.noise_scale
        a=-lam*residual+torch.special.log_ndtr(residual/sigma-lam*sigma)
        b=lam*residual+torch.special.log_ndtr(-residual/sigma-lam*sigma)
        logdensity=math.log(lam/2)+.5*(lam*sigma)**2+torch.logaddexp(a,b)
        return logdensity.sum(-1)
