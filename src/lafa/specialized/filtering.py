"""Online SA-LEnKF for Gaussian transitions and linear Gaussian observations."""
from dataclasses import dataclass,field
import math
import torch
from ..targets import value_and_grad, validate_position
from ._common import OwnedState, checked, positive, noise_like


def _matrix(value,like,shape,name,*,spd=False):
    checked(value,like,name,shape)
    if spd:
        if not torch.allclose(value,value.T,rtol=1e-5,atol=1e-7):
            raise ValueError(f'{name} must be symmetric')
        try:
            torch.linalg.cholesky(value)
        except RuntimeError as exc:
            raise ValueError(f'{name} must be positive definite') from exc
    return value


def lenkf_transition(particles,prior_gradient,observation,H,V,epsilon,*,forecast_noise,observation_noise):
    """R17 Algorithm 2 full observation block (c=1), deterministic kernel.

    Inputs noises are independent standard normal arrays (m,d) and (m,p).
    Returns (analysis, gain). Q=epsilon I, R=2V; forecast uses half drift.
    The initial-prior sign follows Eq. (6), correcting Algorithm 2's sign
    inconsistency: negative potential gradient contracts a Gaussian prior.
    """
    e=positive(epsilon,'epsilon')
    if particles.ndim!=2 or observation.ndim!=1:
        raise ValueError('particles (m,d), observation (p,) required')
    m,d=particles.shape
    p=observation.numel()
    checked(prior_gradient,particles,'prior gradient')
    checked(forecast_noise,particles,'forecast noise')
    checked(observation_noise,particles,'observation noise',(m,p))
    checked(observation,particles,'observation',(p,))
    _matrix(H,particles,(p,d),'linear observation matrix')
    _matrix(V,particles,(p,p),'observation covariance',spd=True)
    innovation=e*(H@H.T)+2*V
    gain=torch.linalg.solve(innovation,e*H).T
    forecast=particles-e/2*prior_gradient+math.sqrt(e)*forecast_noise
    obs_noise=observation_noise@torch.linalg.cholesky(2*V).T
    analysis=forecast+(observation-forecast@H.T-obs_noise)@gain.T
    return analysis,gain


@dataclass
class FilteringResult:
    theta: torch.Tensor
    particles: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor
    stage: int
    stats: dict=field(default_factory=dict)


class SALEnKF(OwnedState):
    """R17 online Algorithm 2, full observation blocks only (n=N).

    transition(states,theta)->states may be nonlinear; process_covariance(theta),
    observation_matrix(theta), observation_covariance(theta) return C,H,V (or
    pass constant tensors). C,V must be SPD; H must be a matrix linear in state.
    prior_potential(states)->(m,) defines the initial prior; initial_particles
    are independent prior samples supplied by the caller. theta is a parameter
    estimate, never advertised as a posterior draw.

    step(observation) completes one stage with inner_steps, retaining ALL m times
    (inner_steps-burnin) post-burnin samples for backward resampling next stage.
    Memory is bounded in number of stages, not in inner_steps. Checkpoints are
    between stages only. epsilon and gamma are constants or callables(stage,k)
    with one-based indices. gamma=0 gives known-parameter filtering. No SA at t=1.
    Finite pools and short inner loops are approximations. Uses negative prior
    potential gradient to resolve the printed prior-score sign inconsistency.
    """
    _state_fields=('_position','_theta','iteration','inner_steps','burnin','_pool')

    def __init__(self,transition,process_covariance,observation_matrix,observation_covariance,prior_potential,initial_particles,theta,*,epsilon=.1,gamma=.001,inner_steps=100,burnin=50,generator=None):
        self._initialize(initial_particles,generator)
        validate_position(theta)
        if initial_particles.ndim!=2 or theta.ndim!=1 or theta.device!=initial_particles.device or theta.dtype!=initial_particles.dtype:
            raise ValueError('initial_particles (m,d) and theta (p,) must share dtype/device')
        if not isinstance(inner_steps,int) or not isinstance(burnin,int) or not 0<=burnin<inner_steps:
            raise ValueError('require integer 0 <= burnin < inner_steps')
        self._theta=theta.detach().clone()
        self.transition=transition
        self.process_covariance=process_covariance
        self.observation_matrix=observation_matrix
        self.observation_covariance=observation_covariance
        self.prior_potential=prior_potential
        self.epsilon=epsilon
        self.gamma=gamma
        self.inner_steps=inner_steps
        self.burnin=burnin
        self._pool=None
        self._rate(epsilon,1,1,positive_only=True)
        self._rate(gamma,1,1,positive_only=False)

    @property
    def theta(self):
        return self._theta.clone()

    @property
    def particles(self):
        return self.position

    @property
    def pool(self):
        return None if self._pool is None else self._pool.clone()

    @staticmethod
    def _rate(schedule,t,k,*,positive_only):
        value=float(schedule(t,k) if callable(schedule) else schedule)
        if not math.isfinite(value) or value<0 or (positive_only and value==0):
            raise ValueError('schedule must return finite positive epsilon or nonnegative gamma')
        return value

    @staticmethod
    def _operator(operator,theta):
        if not callable(operator):
            return operator
        # Keep the graph to theta during predictive differentiation. Snapshot
        # outputs too: separate operators may reuse one callback work buffer.
        value=operator(theta.clone())
        return value.clone() if isinstance(value,torch.Tensor) else value

    def _operators(self,theta,observation):
        d=self._position.shape[1]; p=observation.numel()
        C=self._operator(self.process_covariance,theta)
        H=self._operator(self.observation_matrix,theta)
        V=self._operator(self.observation_covariance,theta)
        _matrix(C,theta,(d,d),'process covariance',spd=True)
        _matrix(H,theta,(p,d),'linear observation matrix')
        _matrix(V,theta,(p,p),'observation covariance',spd=True)
        return C,H,V

    def predictive_score(self,observation,ancestors,theta=None):
        """Mean parameter log-predictive score, including covariance logdet.

        Ancestors are detached: gradients never flow through filtering history
        or categorical ancestor choices. Differentiates all theta-dependent
        operators, including variance parameters, without a Bayesian Jacobian.
        """
        base=self._theta if theta is None else theta
        with torch.enable_grad():
            t=base.detach().clone().requires_grad_()
            C,H,V=self._operators(t,observation)
            means=self.transition(ancestors.detach().clone(),t.clone())
            checked(means,ancestors,'transition means')
            S=H@C@H.T+V
            residual=observation-means@H.T
            terms=.5*(residual*torch.linalg.solve(S,residual.T).T).sum(-1)+.5*torch.linalg.slogdet(S).logabsdet
            gradient=torch.autograd.grad(terms.mean(),t,allow_unused=True)[0] if terms.requires_grad else None
        return torch.zeros_like(base) if gradient is None else checked(-gradient,base,'predictive score')

    def backward_probabilities(self,particles,pool,theta=None,*,block_size=128):
        """Finite-pool Gaussian backward probabilities, rows sum to one.

        Subtract states before whitening to preserve small residuals at large
        common offsets. Row-wise, blocked residuals use O(block_size*d) working
        storage in addition to the returned m by pool probability matrix;
        no m by pool by d array or expanded squared-distance identity is used.
        step requests one current row at a time.
        """
        if not isinstance(block_size,int) or block_size<1:
            raise ValueError('block_size must be a positive integer')
        t=self._theta if theta is None else theta
        with torch.no_grad():
            C=self._operator(self.process_covariance,t)
            _matrix(C,t,(particles.shape[1],particles.shape[1]),'process covariance',spd=True)
            L=torch.linalg.cholesky(C)
            logits=particles.new_empty((len(particles),len(pool)))
            for start in range(0,len(pool),block_size):
                block=pool[start:start+block_size]
                means=self.transition(block.clone(),t.clone())
                checked(means,block,'transition means')
                for index,row in enumerate(particles):
                    residual=means-row
                    whitened=torch.linalg.solve_triangular(L,residual.T,upper=False)
                    logits[index,start:start+len(block)]=-.5*whitened.square().sum(0)
            return logits.softmax(-1)

    def step(self,observation):
        if not isinstance(observation,torch.Tensor) or observation.ndim!=1:
            raise ValueError('observation must be a vector; nonlinear observation callbacks unsupported')
        checked(observation,self._theta,'observation',observation.shape)
        t=self.iteration+1
        theta=self._theta.clone(); x=self._position.clone()
        collection=[]; score_norm=x.new_zeros(())
        ancestors_indices=None
        if t>1:
            with torch.no_grad():
                C,_,_=self._operators(theta,observation)
                # x is already a private forecast buffer and is consumed here.
                means=self.transition(x,theta.clone())
                checked(means,x,'transition means')
                x=means+noise_like(x,self.generator)@torch.linalg.cholesky(C).T
        for k in range(1,self.inner_steps+1):
            epsilon=self._rate(self.epsilon,t,k,positive_only=True)
            gamma=self._rate(self.gamma,t,k,positive_only=False)
            with torch.no_grad():
                C,H,V=self._operators(theta,observation)
                if t>1:
                    indices=[]
                    for row in x.split(1):
                        probs=self.backward_probabilities(row,self._pool,theta)[0]
                        indices.append(torch.multinomial(probs,1,generator=self.generator))
                    ancestors_indices=torch.cat(indices)
                    ancestors=self._pool[ancestors_indices].detach()
                    # Retain the untransformed ancestors for predictive_score.
                    means=self.transition(ancestors.clone(),theta.clone())
                    prior_gradient=torch.linalg.solve(C,(x-means).T).T
                else:
                    _,prior_gradient=value_and_grad(self.prior_potential,x)
                obs_noise=torch.randn((len(x),observation.numel()),device=x.device,dtype=x.dtype,generator=self.generator)
                x,_=lenkf_transition(x,prior_gradient,observation,H,V,epsilon,forecast_noise=noise_like(x,self.generator),observation_noise=obs_noise)
                x=checked(x,self._position,'analysis')
            if k>self.burnin:
                collection.append(x.detach().clone())
            if t>1 and gamma:
                score=self.predictive_score(observation,ancestors,theta)
                theta=checked(theta+gamma*score,theta,'parameter update')
                score_norm=score.norm()
        pool=torch.cat(collection)
        mean=pool.mean(0)
        centered=pool-mean
        covariance=centered.T@centered/max(len(pool)-1,1)
        self._position=x.detach().clone(); self._theta=theta.detach().clone()
        self._pool=pool.detach(); self.iteration=t
        stats={'pool_size':len(pool),'score_norm':score_norm.detach().clone()}
        if ancestors_indices is not None:
            stats['ancestor_indices']=ancestors_indices.clone()
        return FilteringResult(self.theta,self.particles,mean.detach().clone(),covariance.detach().clone(),t,stats)

    def state_dict(self):
        result=super().state_dict()
        # Scalar schedules are serialized. Callable schedules/model operators are
        # caller-owned and must be reconstructed identically on continuation.
        result['schedules']={name:None if callable(getattr(self,name)) else float(getattr(self,name)) for name in ('epsilon','gamma')}
        return result

    def load_state_dict(self,checkpoint):
        checked(checkpoint['state']['_theta'],self._theta,'checkpoint theta')
        super().load_state_dict(checkpoint)
        for name,value in checkpoint['schedules'].items():
            if value is not None:
                setattr(self,name,value)
