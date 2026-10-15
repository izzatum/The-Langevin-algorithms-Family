import math
import pytest
import torch
from lafa.specialized import SALEnKF,lenkf_transition


@pytest.fixture(params=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def device(request):return request.param


def tensor(x,device='cpu'):return torch.tensor(x,dtype=torch.float64,device=device)
def gen(device='cpu',seed=91):return torch.Generator(device=device).manual_seed(seed)


def model(device,*,inner_steps=20,burnin=10,gamma=.001,epsilon=.1,particles=None):
    theta=tensor([.7,math.log(.3),math.log(.4)],device)
    if particles is None:
        particles=torch.randn((16,1),dtype=theta.dtype,device=device,generator=gen(device))
    transition=lambda x,t:t[0]*x
    C=lambda t:t[1].exp().reshape(1,1)
    H=lambda t:t.new_ones((1,1))
    V=lambda t:t[2].exp().reshape(1,1)
    prior=lambda x:x.square().sum(-1)/2
    return SALEnKF(transition,C,H,V,prior,particles,theta,epsilon=epsilon,gamma=gamma,inner_steps=inner_steps,burnin=burnin,generator=gen(device))


def test_scalar_forecast_analysis_half_drift(device):
    x=tensor([[1.],[-2.]],device); g=2*x; y=tensor([.8],device)
    H=tensor([[1.]],device); V=tensor([[.4]],device); e=.15
    z=tensor([[.3],[-.1]],device); z2=tensor([[-.7],[.4]],device)
    result,K=lenkf_transition(x,g,y,H,V,e,forecast_noise=z,observation_noise=z2)
    gain=e/(e+.8)
    forecast=(1-e)*x+math.sqrt(e)*z
    expected=(1-gain)*forecast+gain*y-gain*math.sqrt(.8)*z2
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(K,tensor([[gain]],device))


def test_filter_covariance_identity_and_monte_carlo(device):
    H=tensor([[1.,.4],[-.2,1.]],device); V=tensor([[.7,.1],[.1,.5]],device); e=.2
    x=torch.zeros((25000,2),dtype=H.dtype,device=device); y=x[0]
    generator=gen(device)
    z=torch.randn(x.shape,dtype=x.dtype,device=device,generator=generator)
    z2=torch.randn(x.shape,dtype=x.dtype,device=device,generator=generator)
    result,K=lenkf_transition(x,x,y,H,V,e,forecast_noise=z,observation_noise=z2)
    I=torch.eye(2,dtype=x.dtype,device=device); A=I-K@H
    expected=e*A
    torch.testing.assert_close(e*A@A.T+K@(2*V)@K.T,expected)
    covariance=torch.cov(result.T)
    torch.testing.assert_close(covariance,expected,rtol=.04,atol=.003)


def test_backward_weights_two_gaussian_pool(device):
    f=model(device)
    x=tensor([[.5],[-.2]],device); pool=tensor([[-1.],[1.]],device)
    probabilities=f.backward_probabilities(x,pool)
    expected=torch.softmax(-(x-.7*pool.T).square()/(2*.3),dim=1)
    torch.testing.assert_close(probabilities,expected)


def test_predictive_parameter_derivative_including_logdet(device):
    f=model(device)
    ancestors=tensor([[-1.],[.2],[2.]],device).requires_grad_()
    y=tensor([.8],device)
    score=f.predictive_score(y,ancestors)
    e=y-.7*ancestors.detach().squeeze(-1); s=.7
    expected=torch.stack([(e*ancestors.detach().squeeze(-1)/s).mean(),(.5*.3*(e.square()/s**2-1/s)).mean(),(.5*.4*(e.square()/s**2-1/s)).mean()])
    torch.testing.assert_close(score,expected)
    assert ancestors.grad is None and score.grad_fn is None


def test_filter_stage_state_rng_continuation_and_parameter_update(device):
    f=model(device)
    original=f.theta
    result=f.step(tensor([.4],device))
    torch.testing.assert_close(result.theta,original)
    assert len(f.pool)==16*10
    checkpoint=f.state_dict()
    expected=f.step(tensor([.9],device))
    assert not torch.equal(expected.theta,original)
    second=model(device); second.load_state_dict(checkpoint)
    actual=second.step(tensor([.9],device))
    torch.testing.assert_close(actual.particles,expected.particles,rtol=0,atol=0)
    torch.testing.assert_close(actual.theta,expected.theta,rtol=0,atol=0)
    torch.testing.assert_close(second.pool,f.pool,rtol=0,atol=0)
    assert len(f.pool)==16*10 and f.iteration==2
    saved=second.position
    actual.particles.zero_(); checkpoint['state']['_position'].zero_()
    torch.testing.assert_close(second.position,saved)
    assert second.pool.grad_fn is None


def test_known_parameter_filter_against_short_kalman(device):
    generator=gen(device,101)
    particles=torch.randn((48,1),dtype=torch.float64,device=device,generator=generator)
    f=model(device,inner_steps=90,burnin=40,gamma=0.,epsilon=.08,particles=particles)
    observations=[.4,.9,-.1]
    mean=0.; variance=1.
    for index,y in enumerate(observations):
        if index:
            mean=.7*mean; variance=.7**2*variance+.3
        gain=variance/(variance+.4)
        mean=mean+gain*(y-mean); variance=(1-gain)*variance
        result=f.step(tensor([y],device))
        assert abs(float(result.mean)-mean)<.18
        assert abs(float(result.covariance)-variance)<.12


def test_filter_rejects_nonlinear_observation_and_bad_covariance(device):
    f=model(device)
    f.observation_matrix=lambda t:lambda x:x.square()
    with pytest.raises(ValueError,match='shape'):
        f.step(tensor([.3],device))
    f=model(device); f.process_covariance=tensor([[-1.]],device)
    with pytest.raises(ValueError,match='positive definite'):
        f.step(tensor([.3],device))


def test_schedule_validation_and_checkpoint_scalars(device):
    with pytest.raises(ValueError,match='schedule'):
        model(device,epsilon=-1)
    f=model(device,epsilon=.12,gamma=.02)
    checkpoint=f.state_dict()
    other=model(device,epsilon=.3,gamma=0.)
    other.load_state_dict(checkpoint)
    assert other.epsilon==.12 and other.gamma==.02


@pytest.mark.parametrize('dtype,offset',[(torch.float32,1e4),(torch.float64,1e8)])
def test_backward_weights_preserve_translation_at_large_offsets(device,dtype,offset):
    current=torch.tensor([[.5]],dtype=dtype,device=device)
    pool=torch.tensor([[-1.],[1.]],dtype=dtype,device=device)
    theta=torch.zeros(1,dtype=dtype,device=device)
    covariance=torch.eye(1,dtype=dtype,device=device)
    f=SALEnKF(lambda x,t:x,covariance,covariance,covariance,
             lambda x:x.square().sum(-1)/2,pool,theta,
             inner_steps=2,burnin=1,generator=gen(device))
    centered=f.backward_probabilities(current,pool,block_size=1)
    translated=f.backward_probabilities(current+offset,pool+offset,block_size=1)
    residual=(pool+offset).T-(current+offset)
    expected=(-.5*residual.square()).softmax(-1)
    torch.testing.assert_close(translated,expected)
    torch.testing.assert_close(translated,centered)
    torch.testing.assert_close(translated,theta.new_tensor([[.2689414213699951,.7310585786300049]]))


def test_backward_weights_tiled_correlated_covariance(device):
    theta=tensor([0.],device)
    C=tensor([[2.,.6],[.6,.8]],device)
    H=torch.eye(2,dtype=C.dtype,device=device)
    pool=tensor([[-1.,.2],[.5,-.7],[2.,1.],[-.4,3.]],device)
    current=tensor([[.2,-.4],[1.,2.],[-2.,1.]],device)
    f=SALEnKF(lambda x,t:x,C,H,H,lambda x:x.square().sum(-1)/2,
             pool,theta,inner_steps=2,burnin=1,generator=gen(device))
    # Independent direct Mahalanobis oracle exercises multiple rows and tiles.
    residual=current[:,None,:]-pool[None,:,:]
    logits=-.5*torch.einsum('mnd,de,mne->mn',residual,torch.linalg.inv(C),residual)
    expected=logits.softmax(-1)
    for block_size in (1,3,128):
        torch.testing.assert_close(f.backward_probabilities(current,pool,block_size=block_size),expected)
    offset=tensor([1e8,-1e8],device)
    torch.testing.assert_close(f.backward_probabilities(current+offset,pool+offset,block_size=3),expected,atol=1e-8,rtol=1e-7)
