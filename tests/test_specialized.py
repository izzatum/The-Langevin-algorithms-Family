import math
import pytest
import torch
from lafa.specialized import (AnchoredLangevin,HOLA,MYIPLA,PIPULA,PIPGLA,
    BirthDeathLangevin,gaussian_log_kde,AnnealedLangevin,NoiseAnnealedLangevin,ReplicaExchange)


@pytest.fixture(params=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def device(request):
    return request.param


def tensor(x,device='cpu'):
    return torch.tensor(x,dtype=torch.float64,device=device)


def gen(device='cpu',seed=44):
    return torch.Generator(device=device).manual_seed(seed)


def gaussian(x):
    return x.square().sum(-1)/2


def test_anchored_reference_and_random_clock(device):
    x=tensor([[1.,-2.],[.3,.7]],device); z=x*.2
    sampler=AnchoredLangevin(gaussian,lambda a:gaussian(a)+.4,x,.03,generator=gen(device))
    result=sampler.step(noise=z)
    delta=.03*math.exp(-.4)
    torch.testing.assert_close(result.position,x-delta*x+math.sqrt(2*delta)*z)
    torch.testing.assert_close(sampler.operational_time,torch.full_like(x[:,0],delta))
    result.position.zero_()
    assert not torch.equal(result.position,sampler.position)


def test_anchored_heavy_tail_and_overflow(device):
    x=tensor([1.,2.],device); z=tensor([.4,-.8],device); beta=3.; h=.02
    U=lambda y:(beta+1)*torch.log1p(y.square().sum(-1))
    U0=lambda y:beta*torch.log1p(y.square().sum(-1))
    result=AnchoredLangevin(U,U0,x,h,generator=gen(device)).step(noise=z)
    torch.testing.assert_close(result.position,(1-2*beta*h)*x+math.sqrt(2*h*(1+float(x.square().sum())))*z)
    bad=AnchoredLangevin(lambda y:gaussian(y)+10000,gaussian,x,h)
    with pytest.raises(ValueError,match='overflow'):
        bad.step(noise=z)


def test_hola_quartic_four_tamings(device):
    x=tensor([2.],device); h=.08
    sampler=HOLA(lambda y:(y**4/4+y*y/2).sum(),x,h,generator=gen(device))
    result=sampler.step(noise=torch.zeros_like(x),auxiliary_noise=torch.zeros_like(x))
    g=10.; H=13.; lap=12.
    gh=g/(1+h**1.5*g**1.5)**(2/3)
    c=H*g/(1+h*2*H*g)
    lh=lap/(1+math.sqrt(h)*2*lap)
    torch.testing.assert_close(result.position,tensor([2-h*gh+h*h/2*(c-lh)],device))


def test_hola_quadratic_dense_provider_factorization(device):
    A=tensor([[2.,.4],[.4,1.]],device); x=tensor([.7,-1.],device)
    z=tensor([-.4,.8],device); za=tensor([.2,-.1],device); h=.1
    class Derivatives:
        def gradient(self,x):return A@x
        def hvp(self,x,v):return A@v
        def spectral_norm(self,x):return torch.linalg.eigvalsh(A).abs().max()
        def grad_laplacian(self,x):return torch.zeros_like(x)
    U=lambda y:y@A@y/2
    expected=x-h*A@x+h*h/2*A@A@x+math.sqrt(2*h)*(z-h/2*A@z+h/math.sqrt(12)*A@za)
    for provider in (None,Derivatives()):
        result=HOLA(U,x,h,tamed=False,derivatives=provider,generator=gen(device)).step(noise=z,auxiliary_noise=za)
        torch.testing.assert_close(result.position,expected)
    dense=HOLA(U,x,h,tamed=True).step(noise=z,auxiliary_noise=za)
    provided=HOLA(U,x,h,tamed=True,derivatives=Derivatives()).step(noise=z,auxiliary_noise=za)
    torch.testing.assert_close(dense.position,provided.position)
    I=torch.eye(2,dtype=x.dtype,device=device)
    B=I-h*A/2; D=h*A/math.sqrt(12)
    torch.testing.assert_close(2*h*(B@B.T+D@D.T),2*h*(I-h*A+h*h/3*A@A))


def test_hola_trace_gradient_and_guard(device):
    x=tensor([.3,.7],device)
    U=lambda y:y.square().sum()/2+y.square().sum()**2/4
    sampler=HOLA(U,x,.01)
    g,hvp,norm,lap=sampler._derivatives(x)
    torch.testing.assert_close(lap,8*x)
    with pytest.raises(ValueError,match='dimension guard'):
        HOLA(gaussian,torch.zeros(3,dtype=x.dtype,device=device),max_dense_dimension=2)
    linear=HOLA(lambda y:y.sum(),x,.01,tamed=False)
    torch.testing.assert_close(linear.step(noise=x*0,auxiliary_noise=x*0).position,x-.01)


def coupled_prox(theta,x,lam):
    expanded=theta.expand(len(x),-1)
    mean=(expanded+x)/2
    delta=expanded-x
    new=delta.sign()*(delta.abs()-2*lam*.7).clamp_min(0)
    return mean+new/2,mean-new/2


def smooth(theta,x):
    return ((x-theta)**2).sum(-1)/2+theta.square().sum()/2


@pytest.mark.parametrize('cls',[MYIPLA,PIPGLA])
def test_interacting_joint_prox_order_and_gradient_scaling(device,cls):
    theta=tensor([.5],device); x=tensor([[-1.],[.2],[2.]],device)
    zt=tensor([.4],device); zx=tensor([[.2],[-.7],[.1]],device); h=.03; lam=.2
    sampler=cls(smooth,coupled_prox,theta,x,h,smoothing=lam,generator=gen(device))
    gt=(theta-x).mean(0)+theta; gx=x-theta
    thalf=theta-h*gt+math.sqrt(2*h/len(x))*zt
    xhalf=x-h*gx+math.sqrt(2*h)*zx
    if cls is MYIPLA:
        pt,px=coupled_prox(theta,x,lam)
        expected_t=thalf-h/lam*(theta-pt.mean(0)); expected_x=xhalf-h/lam*(x-px)
    else:
        pt,expected_x=coupled_prox(thalf,xhalf,lam); expected_t=pt.mean(0)
    result=sampler.step(theta_noise=zt,particle_noise=zx)
    torch.testing.assert_close(result.theta,expected_t)
    torch.testing.assert_close(result.particles,expected_x)
    order=torch.tensor([2,0,1],device=device)
    shuffled=cls(smooth,coupled_prox,theta,x[order],h,smoothing=lam).step(theta_noise=zt,particle_noise=zx[order])
    torch.testing.assert_close(shuffled.theta,result.theta)
    torch.testing.assert_close(shuffled.particles,result.particles[order])


def test_pipula_full_gaussian_prox(device):
    A=tensor([[2.,-.5],[-.5,1.]],device); h=.08
    theta=tensor([.6],device); x=tensor([[-1.],[2.]],device)
    def prox(t,x,l):
        v=torch.cat([t.expand(len(x),-1),x],-1)
        sol=torch.linalg.solve(torch.eye(2,dtype=x.dtype,device=x.device)+l*A,v.T).T
        return sol[:,:1],sol[:,1:]
    zt=theta*.3; zx=x*.2
    result=PIPULA(prox,theta,x,h).step(theta_noise=zt,particle_noise=zx)
    p,q=prox(theta,x,h)
    torch.testing.assert_close(result.theta,p.mean(0)+math.sqrt(2*h/2)*zt)
    torch.testing.assert_close(result.particles,q+math.sqrt(2*h)*zx)


def test_birth_death_kde_rates_and_events(device):
    x=tensor([[0.],[4.]],device)
    logs=gaussian_log_kde(x,.7,block_size=1)
    dense=torch.logsumexp(-torch.cdist(x,x).square()/(2*.7**2),1)-math.log(2)-math.log(.7*math.sqrt(2*math.pi))
    torch.testing.assert_close(logs,dense)
    sampler=BirthDeathLangevin(gaussian,x,.1,bandwidth=.7,generator=gen(device))
    rates=sampler.rates()
    torch.testing.assert_close(rates.sum(),rates.new_zeros(()))
    shift=BirthDeathLangevin(lambda x:gaussian(x)+27,x,.1,bandwidth=.7)
    torch.testing.assert_close(rates,shift.rates())
    # Cancel transport drift with prescribed noise. Slot 1 has positive rate:
    # force its death before slot 0; after copy all rates are zero.
    z=.1*x/math.sqrt(.2)
    result=sampler.step(noise=z,uniforms=tensor([1.,0.],device),partners=torch.tensor([1,0],device=device),order=torch.tensor([1,0],device=device))
    torch.testing.assert_close(result.position,torch.zeros_like(x))
    assert result.stats['deaths']==1 and result.stats['births']==0
    assert result.stats['ancestry'].tolist()==[0,0]
    sampler=BirthDeathLangevin(gaussian,x,.1,bandwidth=.7)
    result=sampler.step(noise=z,uniforms=tensor([0.,1.],device),partners=torch.tensor([1,0],device=device))
    assert result.stats['births']==1 and result.stats['deaths']==0
    torch.testing.assert_close(result.position,torch.zeros_like(x))


def test_birth_death_transport_only_and_invalid_partner(device):
    x=tensor([[1.],[2.]],device); z=x*.1
    sampler=BirthDeathLangevin(gaussian,x,.03,reaction_strength=0)
    torch.testing.assert_close(sampler.step(noise=z).position,x-.03*x+math.sqrt(.06)*z)
    with pytest.raises(ValueError,match='another valid slot'):
        sampler.step(partners=torch.tensor([0,1],device=device))


def test_annealing_conventions_and_stage_completion(device):
    x=tensor([1.,2.],device); z=x*.1
    a=AnnealedLangevin(gaussian,x,[2.,1.],.1,steps_per_temperature=1)
    torch.testing.assert_close(a.step(noise=z).position,.9*x+math.sqrt(.4)*z)
    a.step(noise=z)
    with pytest.raises(StopIteration):a.step()
    b=NoiseAnnealedLangevin(lambda x,s:-x/s**2,x,[2.,1.],.1,steps_per_sigma=1)
    torch.testing.assert_close(b.step(noise=z).position,x-.4/2*x/4+math.sqrt(.4)*z)
    assert b.stage==1


def test_replica_exchange_mh_swap_sign(device):
    x=tensor([[3.],[0.]],device)
    sampler=ReplicaExchange(gaussian,x,[1.,3.],.01,generator=gen(device))
    # Reject both local moves. High-energy cold state must swap with probability 1.
    result=sampler.step(noise=torch.zeros_like(x),local_uniforms=tensor([1.,1.],device),swap_uniforms=tensor([.999],device))
    torch.testing.assert_close(result.position,x.flip(0))
    assert result.stats['swapped'].item()
    sampler=ReplicaExchange(gaussian,x.flip(0),[1.,3.],.01)
    result=sampler.step(noise=torch.zeros_like(x),local_uniforms=tensor([1.,1.],device),swap_uniforms=tensor([.5],device))
    assert not result.stats['swapped'].item()


def factories(device):
    x=tensor([.3,-.2],device); particles=tensor([[.2],[-.5],[.8]],device); theta=tensor([.4],device)
    return [lambda:AnchoredLangevin(gaussian,gaussian,x,generator=gen(device)),
            lambda:HOLA(gaussian,x,generator=gen(device)),
            lambda:MYIPLA(smooth,coupled_prox,theta,particles,generator=gen(device)),
            lambda:PIPGLA(smooth,coupled_prox,theta,particles,generator=gen(device)),
            lambda:PIPULA(coupled_prox,theta,particles,generator=gen(device)),
            lambda:BirthDeathLangevin(gaussian,particles,generator=gen(device)),
            lambda:AnnealedLangevin(gaussian,x,[2.,1.],generator=gen(device)),
            lambda:NoiseAnnealedLangevin(lambda x,s:-x,x,[2.,1.],generator=gen(device)),
            lambda:ReplicaExchange(gaussian,particles,[1.,2.,4.],generator=gen(device))]


def test_all_rng_continuation_owned_snapshots_and_graphs(device):
    for factory in factories(device):
        first=factory(); first.step(); checkpoint=first.state_dict()
        expected=first.step()
        second=factory(); second.load_state_dict(checkpoint); actual=second.step()
        attr='particles' if hasattr(expected,'particles') else 'position'
        torch.testing.assert_close(getattr(actual,attr),getattr(expected,attr),rtol=0,atol=0)
        assert getattr(actual,attr).grad_fn is None
        if hasattr(actual,'theta'):
            torch.testing.assert_close(actual.theta,expected.theta,rtol=0,atol=0)
        saved=second.position
        getattr(actual,attr).zero_()
        torch.testing.assert_close(second.position,saved)
        checkpoint['state']['_position'].zero_()
        torch.testing.assert_close(second.position,saved)


@pytest.mark.parametrize('cls',[AnchoredLangevin,HOLA])
def test_finite_initial_validation(cls):
    args=(gaussian,gaussian) if cls is AnchoredLangevin else (gaussian,)
    with pytest.raises(ValueError,match='finite'):
        cls(*args,tensor([float('nan')]))


def test_shared_theta_noise_conditional_covariance():
    # Independent restarts isolate conditional diffusion, avoiding trajectory
    # autocorrelation. Four particles imply theta variance 2h/4, not 2h/16.
    x=tensor([[0.],[0.],[0.],[0.]])
    theta=tensor([0.]); h=.1
    def zero(t,x):return x.sum(-1)*0+t.sum()*0
    def identity(t,x,l):return t.expand(len(x),-1).clone(),x.clone()
    sampler=MYIPLA(zero,identity,theta,x,h,generator=gen(seed=82))
    checkpoint=sampler.state_dict(); theta_samples=[]; latent_samples=[]
    for _ in range(1200):
        # Preserve RNG progression while resetting the deterministic state.
        checkpoint['generator_state']=sampler.generator.get_state()
        sampler.load_state_dict(checkpoint)
        result=sampler.step()
        theta_samples.append(result.theta); latent_samples.append(result.particles)
    theta_variance=torch.stack(theta_samples).var()
    latent_variance=torch.stack(latent_samples).var()
    assert abs(float(theta_variance)-2*h/4)<.006
    assert abs(float(latent_variance)-2*h)<.014


def test_anchored_gaussian_smoothed_laplace_reference():
    mu=.6
    def target(x):return (x.abs()+.5*x*x).sum(-1)
    def reference(x):
        z=x/(math.sqrt(2)*mu)
        return (x*torch.erf(z)+mu*math.sqrt(2/math.pi)*torch.exp(-z*z)+.5*x*x).sum(-1)
    x=tensor([[0.],[1.],[-1.]])
    sampler=AnchoredLangevin(target,reference,x,.02)
    log_scale,gradient=sampler._reference(x)
    torch.testing.assert_close(gradient,x+torch.erf(x/(math.sqrt(2)*mu)))
    torch.testing.assert_close(log_scale[1],log_scale[2])
    assert gradient[0].item()==0
    result=sampler.step(noise=torch.zeros_like(x))
    torch.testing.assert_close(result.position,x-.02*log_scale.exp()[:,None]*gradient)


def test_hola_scalar_stationary_variance_oracle():
    # The quadratic AR(1) invariant variance is q/(1-r²), which differs
    # from continuous target variance 1/a at a finite step.
    a=2.; h=.3
    class Derivatives:
        def gradient(self,x):return a*x
        def hvp(self,x,v):return a*v
        def spectral_norm(self,x):return a
        def grad_laplacian(self,x):return torch.zeros_like(x)
    # Exercise the actual scalar transition with prescribed independent noises.
    sampler=HOLA(lambda x:a*gaussian(x),tensor([1.]),h,tamed=False,derivatives=Derivatives(),generator=gen())
    z=tensor([.4]); za=tensor([-.7])
    r=1-a*h+a*a*h*h/2
    q=2*h*(1-a*h+a*a*h*h/3)
    actual=sampler.step(noise=z,auxiliary_noise=za).position
    expected=r+math.sqrt(2*h)*((1-h*a/2)*z+h*a/math.sqrt(12)*za)
    torch.testing.assert_close(actual,expected)
    assert not math.isclose(q/(1-r*r),1/a,rel_tol=.01)
