import pytest
import torch
from test_core import DT, DEVICES


@pytest.mark.parametrize('device',DEVICES)
def test_optimizer_ula_group_covariance_and_return(device):
    import lafa
    a=torch.nn.Parameter(torch.tensor([1.],dtype=DT,device=device)); b=torch.nn.Parameter(torch.tensor([2.],dtype=DT,device=device))
    seed=22
    opt=lafa.ULA([{'params':[a],'lr':.1},{'params':[b],'lr':.3}], generator=torch.Generator(device=device).manual_seed(seed))
    def closure():
        opt.zero_grad(); loss=(a.square()+b.square()).sum()/2; loss.backward(); return loss
    noise=torch.randn(2,dtype=DT,device=device,generator=torch.Generator(device=device).manual_seed(seed))
    before=torch.tensor([1.,2.],dtype=DT,device=device); h=torch.tensor([.1,.3],dtype=DT,device=device)
    value=opt.step(closure)
    torch.testing.assert_close(torch.cat([a,b]), before-h*before+(2*h).sqrt()*noise)
    torch.testing.assert_close(value,torch.tensor(2.5,dtype=DT,device=device))


@pytest.mark.parametrize('device',DEVICES)
def test_optimizer_mala_joint_rollback_and_checkpoint(device):
    import lafa
    a=torch.nn.Parameter(torch.tensor([1.],dtype=DT,device=device)); b=torch.nn.Parameter(torch.tensor([2.],dtype=DT,device=device))
    opt=lafa.MALA([{'params':[a],'lr':20.},{'params':[b],'lr':30.}],generator=torch.Generator(device=device).manual_seed(3))
    def closure():
        opt.zero_grad(); loss=(a.square()+b.square()).sum()/2; loss.backward(); return loss
    value=opt.step(closure)
    torch.testing.assert_close(torch.cat([a,b]), torch.tensor([1.,2.],dtype=DT,device=device))
    torch.testing.assert_close(torch.cat([a.grad,b.grad]),torch.tensor([1.,2.],dtype=DT,device=device))
    torch.testing.assert_close(value,torch.tensor(2.5,dtype=DT,device=device))
    saved=opt.state_dict()
    opt.step(closure); expected=torch.cat([a,b]).detach().clone()
    opt.load_state_dict(saved); opt.step(closure)
    assert torch.equal(expected,torch.cat([a,b]))

@pytest.mark.parametrize('name',['LipULA','LipMALA','BAOAB'])
@pytest.mark.parametrize('device',DEVICES)
def test_adapters_closure_step(name,device):
    import lafa
    p=torch.nn.Parameter(torch.ones(2,dtype=DT,device=device))
    opt=getattr(lafa,name)([p],lr=.05,generator=torch.Generator(device=device).manual_seed(1))
    def closure():
        opt.zero_grad(); value=p.square().sum()/2;value.backward();return value
    assert opt.step(closure).ndim==0
    assert torch.isfinite(p).all()


def test_optimizer_rejects_aliases_sparse_and_missing_grads():
    import lafa
    p=torch.nn.Parameter(torch.ones(2,dtype=DT))
    with pytest.raises(ValueError): lafa.ULA([p,p])
    q=torch.nn.Parameter(p.detach())
    with pytest.raises(ValueError): lafa.ULA([p,q])
    opt=lafa.ULA([p])
    with pytest.raises(ValueError): opt.step()
    p.grad=torch.sparse_coo_tensor(torch.tensor([[0]]),torch.tensor([1.],dtype=DT),(2,))
    with pytest.raises(ValueError): opt.step()


@pytest.mark.parametrize('device',DEVICES)
def test_optimizer_mala_matches_native_joint_groups(device):
    import lafa
    from lafa.samplers import MALA
    a=torch.nn.Parameter(torch.tensor([1.,-1.],dtype=DT,device=device)); b=torch.nn.Parameter(torch.tensor([2.],dtype=DT,device=device))
    opt=lafa.MALA([{'params':[a],'lr':.15},{'params':[b],'lr':.45}],generator=torch.Generator(device=device).manual_seed(123))
    def potential(x):
        return x.square().sum()/2
    native=MALA(potential,torch.tensor([1.,-1.,2.],dtype=DT,device=device),step_size=.15,
                mobility=torch.tensor([1.,1.,3.],dtype=DT,device=device),generator=torch.Generator(device=device).manual_seed(123))
    def closure():
        opt.zero_grad(); v=potential(torch.cat([a,b])); v.backward(); return v
    for _ in range(10):
        v=opt.step(closure); r=native.step()
        torch.testing.assert_close(torch.cat([a,b]),r.position,rtol=0,atol=0)
        torch.testing.assert_close(v,r.potential,rtol=0,atol=0)
        torch.testing.assert_close(torch.cat([a.grad,b.grad]),r.gradient,rtol=0,atol=0)


@pytest.mark.parametrize('name',['ULA','MALA','LipULA','LipMALA','BAOAB'])
@pytest.mark.parametrize('device',DEVICES)
def test_optimizer_checkpoint_new_instance(name,device):
    import lafa
    p=torch.nn.Parameter(torch.ones(2,dtype=DT,device=device))
    opt=getattr(lafa,name)([p],lr=.1,generator=torch.Generator(device=device).manual_seed(52))
    def closure():
        opt.zero_grad(); v=p.square().sum()/2; v.backward(); return v
    for _ in range(3): opt.step(closure)
    saved=opt.state_dict()
    expected=[]
    for _ in range(5):
        value=opt.step(closure)
        expected.append((p.detach().clone(),p.grad.detach().clone(),value.detach().clone()))
    q=torch.nn.Parameter(torch.zeros(2,dtype=DT,device=device))
    other=getattr(lafa,name)([q],lr=.05,generator=torch.Generator(device=device).manual_seed(999))
    other.load_state_dict(saved)
    def other_closure():
        other.zero_grad(); v=q.square().sum()/2; v.backward(); return v
    for position,gradient,value in expected:
        resumed_value=other.step(other_closure)
        assert q.device.type==device and q.dtype==DT
        assert torch.equal(q,position)
        assert torch.equal(q.grad,gradient)
        assert torch.equal(resumed_value,value)


@pytest.mark.parametrize('name',['ULA','MALA','LipULA','LipMALA','BAOAB'])
@pytest.mark.parametrize('alias',[False,True])
def test_dynamic_parameter_groups_rejected_before_mutation(name,alias):
    import lafa
    p=torch.nn.Parameter(torch.ones(2,dtype=DT))
    opt=getattr(lafa,name)([p],generator=torch.Generator().manual_seed(8))
    q=torch.nn.Parameter(p.detach() if alias else torch.zeros_like(p))
    before=opt.generator.get_state().clone()
    with pytest.raises(ValueError,match='after initialization'):
        opt.add_param_group({'params':[q],'lr':.1})
    assert len(opt.param_groups)==1
    assert torch.equal(before,opt.generator.get_state())
    assert torch.equal(p,torch.ones_like(p))


@pytest.mark.parametrize('name',['MALA','LipMALA'])
def test_optimizer_nonfinite_proposal_rejected_before_closure(name):
    import lafa
    from test_core import validated_normal_potential
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float32))
    kwargs={'max_step_size':1e30} if name=='LipMALA' else {}
    opt=getattr(lafa,name)([p],lr=1e30,temperature=1e30,generator=torch.Generator().manual_seed(3),**kwargs)
    def closure():
        assert torch.isfinite(p).all(), 'closure saw invalid proposal coordinates'
        opt.zero_grad(); value=validated_normal_potential(p); value.backward(); return value
    expected=validated_normal_potential(p).detach()
    value=opt.step(closure)
    assert not opt.last_result.accepted
    torch.testing.assert_close(p,torch.zeros_like(p))
    torch.testing.assert_close(p.grad,torch.zeros_like(p))
    torch.testing.assert_close(value,expected)
    if name=='LipMALA':
        torch.testing.assert_close(opt._sampler.step_size,torch.tensor(1e30,dtype=p.dtype))
        assert torch.isposinf(opt._sampler._alpha)
