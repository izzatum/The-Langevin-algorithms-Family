"""SALEnKF work per stage: one pool transition and one process factor per inner step."""
import math
import pytest
import torch
from lafa.specialized import SALEnKF


def tensor(x):return torch.tensor(x,dtype=torch.float64)
def gen(seed=91):return torch.Generator().manual_seed(seed)
def prior(x):return x.square().sum(-1)/2
def transition(x,t):return torch.tanh(t[0]*x@x.new_tensor([[.8,.1],[-.2,.7]]).T)+t[1]*torch.sin(x)
def covariance(t):return t[2].exp()*t.new_tensor([[.5,.1],[.1,.3]])


class Counted:
    """Wrap a deterministic callback; record first-argument length and theta per call."""
    def __init__(self,function):self.function=function;self.reset()
    def reset(self):self.rows,self.thetas=[],[]
    def __call__(self,*arguments):
        self.rows.append(len(arguments[0])); self.thetas.append(arguments[-1].detach().clone())
        return self.function(*arguments)


def sampler(m,inner_steps,burnin,gamma,*,moves=transition,process=covariance,seed=5):
    """d=2, p=1: nonlinear transition and theta-dependent dense process covariance."""
    generator=gen(seed)
    theta=tensor([.9,.2,-.4])
    particles=torch.randn((m,2),dtype=theta.dtype,generator=generator)
    moves,process=Counted(moves),Counted(process)
    f=SALEnKF(moves,process,tensor([[1.,-.5]]),tensor([[.3]]),prior,particles,theta,
              epsilon=.1,gamma=gamma,inner_steps=inner_steps,burnin=burnin,generator=generator)
    return f,moves,process


def distinct(thetas):return len({tuple(theta.tolist()) for theta in thetas})


# (10,10,5) is the audited case: 610 row-evaluations, 5110 before. (6,30,5) spans two pool blocks.
@pytest.mark.parametrize('m,inner_steps,burnin',[(10,10,5),(6,30,5)])
def test_known_parameter_stage_transitions_pool_once_per_inner_step(m,inner_steps,burnin):
    f,moves,process=sampler(m,inner_steps,burnin,0.)
    f.step(tensor([.4]))
    pool=len(f.pool)
    assert moves.rows==[] and pool==m*(inner_steps-burnin)
    process.reset()
    f.step(tensor([.9]))
    # Forecast m, then per inner step the pool once and the m drawn ancestors.
    assert sum(moves.rows)<=m+inner_steps*(pool+m)
    assert len(moves.rows)<=1+inner_steps*(math.ceil(pool/128)+1)
    assert len(process.rows)<=1+inner_steps


def test_parameter_learning_stage_adds_one_ancestor_transition_per_inner_step():
    m,inner_steps,burnin=10,10,5
    f,moves,process=sampler(m,inner_steps,burnin,.002)
    f.step(tensor([.4]))
    pool=len(f.pool); original=f.theta
    process.reset()
    result=f.step(tensor([.9]))
    # Measured bound for gamma>0: the predictive score transitions the ancestors once more.
    assert sum(moves.rows)<=m+inner_steps*(pool+2*m)
    assert len(process.rows)<=1+2*inner_steps
    # theta moves every inner step, so the pool is transitioned once at each value.
    pooled=[theta for rows,theta in zip(moves.rows,moves.thetas) if rows==pool]
    assert len(pooled)==distinct(pooled)==distinct(moves.thetas)==inner_steps
    assert not torch.equal(result.theta,original)


def test_backward_draws_use_the_process_factor_of_the_current_theta(monkeypatch):
    m,inner_steps=4,6
    f,moves,_=sampler(m,inner_steps,2,.05)
    f.step(tensor([.4]))
    pool=len(f.pool)
    assert pool<=128  # one block, hence one solve per row per inner step
    moves.reset()
    real=torch.linalg.solve_triangular; factors=[]
    def solve(factor,*arguments,**keywords):
        factors.append(factor.clone())
        return real(factor,*arguments,**keywords)
    monkeypatch.setattr(torch.linalg,'solve_triangular',solve)
    f.step(tensor([.9]))
    thetas=[theta for rows,theta in zip(moves.rows,moves.thetas) if rows==pool]
    assert len(thetas)==distinct(thetas)==inner_steps and len(factors)==m*inner_steps
    # theta moves every inner step; a factor kept from an earlier theta would be stale.
    for k,theta in enumerate(thetas):
        expected=torch.linalg.cholesky(covariance(theta))
        assert all(torch.equal(used,expected) for used in factors[k*m:(k+1)*m])


def test_process_covariance_is_factorised_once_per_validation(monkeypatch):
    m,inner_steps=5,6
    f,_,process=sampler(m,inner_steps,2,0.)
    f.step(tensor([.4]))
    process.reset()
    real=torch.linalg.cholesky; shapes=[]
    def cholesky(matrix,*arguments,**keywords):
        shapes.append(tuple(matrix.shape))
        return real(matrix,*arguments,**keywords)
    monkeypatch.setattr(torch.linalg,'cholesky',cholesky)
    f.step(tensor([.9]))
    # d=2 process factors are told apart from the p=1 observation ones by shape:
    # one validating factorisation for the forecast and one per inner step.
    assert shapes.count((2,2))==len(process.rows)==1+inner_steps


def test_process_covariance_is_validated_at_every_inner_step():
    initial=tensor([.9,.2,-.4])
    # SPD only at the initial theta: the first parameter update must be caught.
    f,_,process=sampler(4,6,2,.002,process=lambda t:covariance(t) if torch.equal(t,initial) else -covariance(t))
    f.step(tensor([.4]))
    theta,position,pool=f.theta,f.position,f.pool
    process.reset()
    with pytest.raises(ValueError,match='process covariance must be positive definite'):
        f.step(tensor([.9]))
    # Forecast, first inner step and its score pass; the second inner step fails.
    assert len(process.rows)==4 and distinct(process.thetas)==2
    assert torch.equal(f.theta,theta) and torch.equal(f.position,position) and torch.equal(f.pool,pool)
    assert f.iteration==1
    f,_,_=sampler(4,6,2,0.,process=lambda t:t.new_tensor([[.5,.1],[.3,.4]]))
    with pytest.raises(ValueError,match='process covariance must be symmetric'):
        f.backward_probabilities(position,pool)
    with pytest.raises(ValueError,match='process covariance must be symmetric'):
        f.step(tensor([.4]))


def test_backward_probabilities_match_dense_gaussian_formula():
    f,moves,_=sampler(4,3,1,0.)
    generator=gen(17)
    pool=torch.randn((150,2),dtype=torch.float64,generator=generator)
    current=torch.randn((5,2),dtype=torch.float64,generator=generator)
    for theta in (None,tensor([.6,-.3,.2])):
        t=f.theta if theta is None else theta
        C=covariance(t)
        residual=current[:,None,:]-transition(pool,t)[None,:,:]
        quadratic=torch.einsum('mnd,de,mne->mn',residual,torch.linalg.inv(C),residual)
        density=torch.exp(-.5*quadratic)/torch.sqrt((2*math.pi)**2*torch.linalg.det(C))
        expected=density/density.sum(-1,keepdim=True)
        for block_size in (1,7,128,1000):
            moves.reset()
            actual=f.backward_probabilities(current,pool,theta,block_size=block_size)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(actual.sum(-1),torch.ones(5,dtype=torch.float64))
            assert actual.shape==(5,150) and (actual>=0).all()
            # Every pool state is transitioned once, whatever the number of rows.
            assert sum(moves.rows)==150 and len(moves.rows)==math.ceil(150/block_size)
        torch.testing.assert_close(f.backward_probabilities(current,pool,theta),expected)
        torch.testing.assert_close(f.backward_probabilities(current[2:3],pool,theta)[0],expected[2])


def test_step_draws_one_ancestor_per_row_from_backward_probabilities():
    m=6
    f,_,_=sampler(m,30,5,0.)
    f.step(tensor([.4]))
    f.inner_steps,f.burnin=1,0
    theta,pool=f.theta,f.pool
    assert len(pool)==150
    twin=gen(); twin.set_state(f.generator.get_state())
    noise=torch.randn((m,2),dtype=theta.dtype,generator=twin)
    forecast=transition(f.position,theta)+noise@torch.linalg.cholesky(covariance(theta)).T
    expected=torch.cat([torch.multinomial(f.backward_probabilities(row,pool,theta)[0],1,generator=twin) for row in forecast.split(1)])
    result=f.step(tensor([.9]))
    assert torch.equal(result.stats['ancestor_indices'],expected)
    assert len(expected.unique())>1


def test_reused_transition_work_buffer_does_not_alias_stored_pool_means():
    buffer=torch.empty((512,2),dtype=torch.float64)
    def reusing(x,t):
        buffer[:len(x)]=transition(x,t)
        return buffer[:len(x)]
    outcomes=[]
    for moves in (transition,reusing):
        f,_,_=sampler(6,30,5,0.,moves=moves)
        results=[f.step(tensor([y])) for y in (.4,.9)]
        outcomes.append((results[-1].particles,results[-1].stats['ancestor_indices'],f.pool))
    # The 150-state pool spans two transition blocks that share the callback buffer.
    for expected,actual in zip(*outcomes):
        assert torch.equal(actual,expected)


def test_invalid_pool_transition_means_fail_before_state_changes():
    m=4
    f,moves,_=sampler(m,6,2,0.)
    f.step(tensor([.4]))
    theta,position,pool=f.theta,f.position,f.pool
    moves.function=lambda x,t:transition(x,t) if len(x)==m else torch.full_like(x,math.nan)
    with pytest.raises(ValueError,match='transition means'):
        f.step(tensor([.9]))
    assert torch.equal(f.theta,theta) and torch.equal(f.position,position) and torch.equal(f.pool,pool)
    assert f.iteration==1


@pytest.mark.parametrize('inner_steps,burnin',[(True,0),(True,False),(2,True),(5,False)])
def test_boolean_inner_steps_and_burnin_are_rejected(inner_steps,burnin):
    with pytest.raises(ValueError,match='integer 0 <= burnin < inner_steps'):
        sampler(3,inner_steps,burnin,0.)


@pytest.mark.parametrize('inner_steps,burnin',[(2.,1),(3,1.),('3',1),(3,None),(3,3),(3,-1),(0,0)])
def test_non_integer_or_unordered_inner_steps_and_burnin_stay_rejected(inner_steps,burnin):
    with pytest.raises(ValueError,match='integer 0 <= burnin < inner_steps'):
        sampler(3,inner_steps,burnin,0.)
    f,_,_=sampler(3,1,0,0.)
    assert (f.inner_steps,f.burnin)==(1,0) and len(f.step(tensor([.4])).particles)==3
