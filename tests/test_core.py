import math
import pytest
import torch

DT = torch.float64
DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]

def api():
    import lafa
    return lafa

def gaussian(x):
    return x.square().sum(-1) / 2

@pytest.mark.parametrize('device', DEVICES)
def test_ula_equation_and_gradient_ownership(device):
    from lafa.functional import ula_step
    from lafa.targets import value_and_grad
    x = torch.tensor([1., -2.], dtype=DT, device=device, requires_grad=True)
    v, g = value_and_grad(gaussian, x)
    z = torch.tensor([.4, -.8], dtype=DT, device=device)
    m = torch.tensor([1., 4.], dtype=DT, device=device)
    y = ula_step(x.detach(), g, .2, temperature=2., mobility=m, noise=z)
    torch.testing.assert_close(y, x.detach() - .2*m*x.detach() + (0.8*m).sqrt()*z)
    assert x.grad is None and not v.requires_grad and not g.requires_grad

def test_mala_ratio_reverse_density():
    from lafa.functional import mala_log_acceptance
    x, y = torch.tensor([1.], dtype=DT), torch.tensor([2.], dtype=DT)
    h, t, m = .3, 1.7, 2.
    got = mala_log_acceptance(x, y, gaussian(x), gaussian(y), x, y, h, temperature=t, mobility=m)
    f = torch.distributions.Normal(x-h*m*x, math.sqrt(2*h*t*m)).log_prob(y).sum()
    r = torch.distributions.Normal(y-h*m*y, math.sqrt(2*h*t*m)).log_prob(x).sum()
    torch.testing.assert_close(got, (gaussian(x)-gaussian(y))/t+r-f)

@pytest.mark.parametrize('name', ['ULA','MALA','LipULA','LipMALA','BAOAB','KineticLangevin'])
def test_checkpoint_exact_continuation_and_snapshot(name):
    import lafa.samplers as samplers
    cls = getattr(samplers, name)
    s = cls(gaussian, torch.ones(3, dtype=DT), step_size=.1, generator=torch.Generator().manual_seed(81))
    result = s.step()
    before = result.position.clone()
    checkpoint = s.state_dict()
    expected = [s.step().position for _ in range(4)]
    other = cls(gaussian, torch.zeros(3, dtype=DT), step_size=.5)
    other.load_state_dict(checkpoint)
    for target in expected:
        assert torch.equal(target, other.step().position)
    assert torch.equal(before, result.position)


def test_mala_independent_chains_and_rejection_gradient():
    from lafa.samplers import MALA
    x = torch.ones(100, 2, dtype=DT)
    s = MALA(gaussian, x, step_size=2., generator=torch.Generator().manual_seed(10))
    r = s.step()
    assert r.accepted.any() and (~r.accepted).any()
    torch.testing.assert_close(r.position[~r.accepted], x[~r.accepted])
    torch.testing.assert_close(r.gradient, r.position)
    assert r.accepted.shape == (100,)


def test_nonfinite_proposal_rejected_and_initial_validation():
    from lafa.samplers import MALA, ULA
    def bounded(x):
        return torch.where(x.abs().sum(-1) < .01, gaussian(x), torch.full_like(x.sum(-1), float('inf')))
    s = MALA(bounded, torch.zeros(2, dtype=DT), step_size=1.)
    r = s.step()
    assert not r.accepted and torch.equal(r.position, torch.zeros(2, dtype=DT))
    with pytest.raises(ValueError):
        ULA(gaussian, torch.tensor([float('nan')]))
    with pytest.raises(ValueError):
        ULA(lambda x: x.sum(), torch.ones(2, 3))
    with pytest.raises(ValueError):
        ULA(gaussian, torch.ones(2), mobility=-1.)


def test_ula_discrete_gaussian_covariance():
    from lafa.samplers import ULA
    s = ULA(gaussian, torch.zeros(20000, 1, dtype=DT), step_size=.3, generator=torch.Generator().manual_seed(5))
    for _ in range(55):
        r = s.step()
    expected = 1 / (1 - .3/2)
    assert abs(r.position.var().item() - expected) < .04


def test_lip_degenerate_freeze_and_rejection():
    from lafa.samplers import LipULA, LipMALA
    zero = lambda x: x.sum(-1)*0
    s = LipULA(zero, torch.zeros(8, 2, dtype=DT), step_size=.1, max_step_size=.4, adapt_steps=3)
    for _ in range(3): s.step()
    h = s.step_size.clone()
    for _ in range(4): s.step()
    assert torch.equal(h, s.step_size) and torch.isfinite(h).all() and (h <= .4).all()
    def bounded(x):
        return torch.where(x.abs().sum(-1)<.001, gaussian(x), torch.full_like(x.sum(-1), float('inf')))
    s = LipMALA(bounded, torch.zeros(3, dtype=DT), step_size=.1)
    h = s.step_size.clone()
    for _ in range(3): assert not s.step().accepted
    assert torch.equal(h, s.step_size)


def test_baoab_exact_ou_and_deterministic_limit():
    from lafa.samplers import BAOAB
    zero = lambda x: x.sum(-1)*0
    s = BAOAB(zero, torch.zeros(25000, 1, dtype=DT), step_size=.3, damping=2., mass=3., temperature=1.4,
              generator=torch.Generator().manual_seed(12))
    s.step()
    assert abs(s.momentum.var().item() - 3*1.4*(1-math.exp(-1.2))) < .07
    s = BAOAB(gaussian, torch.tensor([2.], dtype=DT), step_size=.2, damping=0., momentum=torch.tensor([1.], dtype=DT))
    r=s.step()
    torch.testing.assert_close(r.position, torch.tensor([2.16], dtype=DT))
    torch.testing.assert_close(s.momentum, torch.tensor([.584], dtype=DT))


def test_target_adapters_and_module():
    from lafa.targets import from_log_prob, from_score, positive_transform, value_and_grad, ModulePotential
    from lafa.samplers import ULA, MALA
    x=torch.tensor([.2,-.4],dtype=DT)
    torch.testing.assert_close(value_and_grad(from_log_prob(lambda x: -gaussian(x)),x)[1],x)
    transformed=positive_transform(lambda z: z.sum(-1))
    v,g=value_and_grad(transformed,x)
    torch.testing.assert_close(v,x.exp().sum()-x.sum())
    torch.testing.assert_close(g,x.exp()-1)
    score=from_score(lambda x:-x)
    ULA(score,x).step()
    with pytest.raises(ValueError): MALA(score,x)
    model=torch.nn.Linear(2,1,bias=False,dtype=DT)
    target=ModulePotential(model, lambda output: output.square().sum()/2, args=(torch.ones(1,2,dtype=DT),))
    flat=target.flatten_parameters()
    v,g=value_and_grad(target,flat)
    assert g.shape==flat.shape and len(target.parameter_spec)==1
    assert all(p.grad is None for p in model.parameters())


def test_lip_initial_infinite_ratio_and_exact_recurrence():
    from lafa.samplers import LipULA
    x=torch.tensor([1.,2.],dtype=DT)
    m=torch.tensor([2.,3.],dtype=DT)
    s=LipULA(gaussian,x,step_size=.01,mobility=m,lc=.7,max_step_size=.9,generator=torch.Generator().manual_seed(4))
    r=s.step()
    dx=r.position-x
    expected=.7*torch.linalg.vector_norm(dx)/torch.linalg.vector_norm(m*dx)
    torch.testing.assert_close(s.step_size,expected)
    old_h=s.step_size.clone(); old_x=s.position
    r=s.step()
    dx=r.position-old_x
    bound=.7*torch.linalg.vector_norm(dx)/torch.linalg.vector_norm(m*dx)
    expected=torch.minimum(torch.sqrt(1+old_h/.01)*old_h,bound).clamp_max(.9)
    torch.testing.assert_close(s.step_size,expected)


def test_public_snapshot_mutation_does_not_change_sampler():
    from lafa.samplers import MALA
    s=MALA(gaussian,torch.ones(2,dtype=DT))
    r=s.step(); expected=s.position
    r.position.zero_(); r.gradient.fill_(100); s.position.zero_()
    assert torch.equal(expected,s.position)
    checkpoint=s.state_dict(); checkpoint['state']['_position'].zero_()
    assert torch.equal(expected,s.position)


def test_no_grad_evaluation_and_constant_leaf_potential():
    from lafa.targets import value_and_grad
    x=torch.ones(2,dtype=DT)
    with torch.no_grad():
        v,g=value_and_grad(gaussian,x)
    torch.testing.assert_close(g,x)
    v,g=value_and_grad(lambda x: torch.tensor(2.,dtype=x.dtype,requires_grad=True),x)
    torch.testing.assert_close(g,torch.zeros_like(x))

@pytest.mark.parametrize('name',['ULA','MALA','LipULA','LipMALA','BAOAB','KineticLangevin'])
@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_sampler_device_dtype(name,device,dtype):
    import lafa.samplers as samplers
    s=getattr(samplers,name)(gaussian,torch.ones(3,2,device=device,dtype=dtype),step_size=.05,
                            generator=torch.Generator(device=device).manual_seed(7))
    result=s.step()
    assert result.position.device.type==device and result.position.dtype==dtype
    assert not result.gradient.requires_grad and torch.isfinite(result.position).all()


@pytest.mark.parametrize('score_only',[True,False])
def test_protocol_target_cannot_mutate_caller_or_sampler_position(score_only):
    from lafa.targets import value_and_grad, from_score
    from lafa.samplers import ULA
    class MutatingTarget:
        def value_and_grad(self,x):
            original=x.clone()
            x.neg_()
            return gaussian(original),original
    target=from_score(lambda x:x.neg_()) if score_only else MutatingTarget()
    x=torch.tensor([1.,2.],dtype=DT)
    before=x.clone()
    _,g=value_and_grad(target,x)
    assert torch.equal(x,before)
    torch.testing.assert_close(g,before)
    s=ULA(target,x)
    assert torch.equal(x,before) and torch.equal(s.position,before)
    s.invalidate_cache()
    assert torch.equal(s.position,before)
    torch.testing.assert_close(s._gradient,before)


def validated_normal_potential(x):
    return -torch.distributions.Normal(torch.zeros_like(x),torch.ones_like(x),validate_args=True).log_prob(x).sum(-1)


@pytest.mark.parametrize('name',['MALA','LipMALA'])
@pytest.mark.parametrize('mixed',[False,True])
def test_nonfinite_coordinates_rejected_before_target_evaluation(name,mixed):
    import lafa.samplers as samplers
    x=torch.zeros(2,1,dtype=torch.float32) if mixed else torch.zeros(1,dtype=torch.float32)
    h=torch.tensor([1e30,.01],dtype=x.dtype) if mixed else 1e30
    m=torch.tensor([[1e30],[1.]],dtype=x.dtype) if mixed else 1e30
    kwargs={'max_step_size':1e30} if name=='LipMALA' else {}
    s=getattr(samplers,name)(validated_normal_potential,x,step_size=h,mobility=m,
                            generator=torch.Generator().manual_seed(2),**kwargs)
    before=s.state_dict()['state']
    r=s.step()
    bad=~r.accepted
    assert bool(bad[0]) if mixed else bool(bad)
    if mixed:
        assert bool(r.accepted[1])
        assert not torch.equal(r.position[1],x[1])
    torch.testing.assert_close(r.position[bad],x[bad])
    torch.testing.assert_close(r.gradient[bad],before['_gradient'][bad])
    torch.testing.assert_close(r.potential[bad],before['_value'][bad])
    if name=='LipMALA':
        assert torch.equal(s.step_size[bad],before['step_size'][bad])
        assert torch.equal(s._alpha[bad],before['_alpha'][bad])
    assert torch.isneginf(r.stats['log_acceptance'][bad]).all()


@pytest.mark.parametrize('device',DEVICES)
def test_module_potential_rejects_distinct_parameters_sharing_storage(device):
    from lafa.targets import ModulePotential
    class AliasedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p=torch.nn.Parameter(torch.ones(1,dtype=DT,device=device))
            self.q=torch.nn.Parameter(self.p.detach())
        def forward(self):
            return self.p-self.q
    model=AliasedModule()
    assert torch.equal(model(),torch.zeros(1,dtype=DT,device=device))
    with pytest.raises(ValueError,match='aliased parameter storage'):
        ModulePotential(model,lambda output:output.square().sum())
    assert model.p is not model.q
    assert torch.equal(model.p,torch.ones_like(model.p))


@pytest.mark.parametrize('device',DEVICES)
def test_module_potential_preserves_same_object_tied_parameters(device):
    from lafa.targets import ModulePotential, value_and_grad
    class TiedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p=torch.nn.Parameter(torch.ones(1,dtype=DT,device=device))
            self.q=self.p
        def forward(self):
            return self.p+self.q
    model=TiedModule()
    target=ModulePotential(model,lambda output:output.square().sum()/2)
    assert target.numel==1 and len(target.parameter_spec)==1
    torch.testing.assert_close(target.flatten_parameters(),torch.ones(1,dtype=DT,device=device))
    x=torch.tensor([3.],dtype=DT,device=device)
    value,gradient=value_and_grad(target,x)
    torch.testing.assert_close(value,torch.tensor(18.,dtype=DT,device=device))
    torch.testing.assert_close(gradient,torch.tensor([12.],dtype=DT,device=device))
    assert model.p is model.q
    torch.testing.assert_close(model(),torch.tensor([2.],dtype=DT,device=device))
    assert model.p.grad is None
