import math
import pytest
import torch
from lafa.kernels import RBF, IMQ
from lafa.particles import (direction, SVGD, WeightSpaceSVGD, ProjectedSVGD,
    SteinVariationalNewton, FunctionSpaceSVGD, RepulsiveEnsemble)

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]
def normal(x): return x.square().sum(-1)/2

def pair_reference(x,s,kernel,w):
    out = torch.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(x)):
            source = x[j].clone().requires_grad_()
            t = ((source-x[i])**2).sum()
            k = torch.exp(-t/(2*kernel.length_scale**2)) if isinstance(kernel,RBF) else (kernel.scale**2+t)**kernel.beta
            grad = torch.autograd.grad(k,source)[0]
            out[i] += w[j]*(k.detach()*s[j]+grad)
    return out

@pytest.mark.parametrize('device', DEVICES)
def test_svgd_sign_single_and_alias(device):
    x = torch.tensor([[-1.],[1.]],dtype=torch.float64,device=device)
    q = math.exp(-2)
    torch.testing.assert_close(direction(x,torch.zeros_like(x),kernel=RBF()), x*q)
    torch.testing.assert_close(direction(x,-x,kernel=RBF()),x*((-1+3*q)/2))
    torch.testing.assert_close(direction(x[:1],-x[:1],kernel=RBF()),-x[:1])
    assert WeightSpaceSVGD is SVGD

@pytest.mark.parametrize('kernel',[RBF(.8),IMQ(.8,-.4)])
@pytest.mark.parametrize('block_size',[1,2,4,20])
def test_direction_pair_reference(kernel,block_size):
    x = torch.randn(5,3,dtype=torch.float64,generator=torch.Generator().manual_seed(19))
    w = torch.arange(1,6,dtype=x.dtype); w/=w.sum()
    torch.testing.assert_close(direction(x,-x,kernel=kernel,weights=w,block_size=block_size),pair_reference(x,-x,kernel,w))

@pytest.mark.parametrize('device',DEVICES)
def test_svn_one_particle_gaussian(device):
    a = torch.tensor([[2.,.4],[.4,1.]],dtype=torch.float64,device=device)
    mean = torch.tensor([.3,-.2],dtype=a.dtype,device=device)
    potential = lambda x: ((x-mean)@a*(x-mean)).sum(-1)/2
    obj = SteinVariationalNewton(potential,mean[None]+2,step_size=1.,damping=0.,cg_tolerance=1e-11)
    result = obj.step()
    torch.testing.assert_close(result.position,mean[None],atol=1e-10,rtol=0)
    assert obj.approximation=='block_diagonal'
    assert result.stats['cg_relative_residual'].max()<1e-9

def test_svn_two_particle_exact():
    x = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    q = math.exp(-2)
    field = x*((1-q)*(-1+3*q)/(1+5*q*q))
    method = SteinVariationalNewton(normal,x,step_size=.2,damping=0,cg_tolerance=1e-12)
    torch.testing.assert_close(method.step().position,x+.2*field)

def test_svn_negative_curvature_rejected():
    method = SteinVariationalNewton(lambda x:-normal(x),torch.tensor([[1.]],dtype=torch.float64),damping=0)
    with pytest.raises(RuntimeError,match='curvature'): method.step()

def test_projected_likelihood_span_and_complement():
    x = torch.randn(7,3,dtype=torch.float64,generator=torch.Generator().manual_seed(11))
    axis = torch.tensor([1.,2.,-1.],dtype=x.dtype); axis/=axis.norm()
    ll = lambda x: -.5*(x@axis-1).square()
    obj = ProjectedSVGD(ll,x,rank=1,step_size=.1)
    torch.testing.assert_close(obj.basis.T@obj.basis,torch.eye(1,dtype=x.dtype))
    torch.testing.assert_close(obj.basis@obj.basis.T,axis[:,None]*axis[None,:])
    complement = obj.complement.clone()
    obj.step();obj.step()
    torch.testing.assert_close(obj.complement,complement,atol=0,rtol=0)
    torch.testing.assert_close(obj.position@(torch.eye(3,dtype=x.dtype)-axis[:,None]*axis[None,:]),complement)

def test_projected_full_rank_matches_svgd():
    x = torch.randn(6,3,dtype=torch.float64,generator=torch.Generator().manual_seed(23))
    ll = lambda x:-((x-1)*torch.tensor([1.,2.,3.],dtype=x.dtype)).square().sum(-1)/2
    obj = ProjectedSVGD(ll,x,rank=3,step_size=.03,kernel=RBF(.7))
    ref = SVGD(lambda x:normal(x)-ll(x),x,step_size=.03,kernel=RBF(.7))
    torch.testing.assert_close(obj.step().position,ref.step().position)

@pytest.mark.parametrize('device',DEVICES)
def test_function_destination_vjp(device):
    x = torch.tensor([[-.7,.4],[1.2,-.3]],dtype=torch.float64,device=device)
    fmap = lambda t:torch.stack([t[0]**2+t[1],2*t[1]])
    z = torch.stack([fmap(t) for t in x])
    phi = pair_reference(z,-z,RBF(.8),x.new_full((2,),.5))
    expected = torch.stack([torch.tensor([[2*t[0],0.],[1.,2.]],device=device,dtype=x.dtype)@v for t,v in zip(x,phi)])
    obj = FunctionSpaceSVGD(fmap,normal,x,step_size=.1,kernel=RBF(.8))
    torch.testing.assert_close(obj.step().position,x+.1*expected)

def test_function_linear_pullback_metric():
    x = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    z = 2*x
    phi = pair_reference(z,-z,RBF(),x.new_full((2,),.5))
    obj = FunctionSpaceSVGD(lambda t:2*t,normal,x,step_size=.1)
    torch.testing.assert_close(obj.step().position,x+.2*phi)

def test_repulsive_ensemble_distinct_and_duplicates():
    x = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    q = math.exp(-2)
    obj = RepulsiveEnsemble(normal,x,step_size=.1)
    expected = -x+2*x*q/(1+q)
    torch.testing.assert_close(obj.step().position,x+.1*expected)
    duplicates = torch.ones(3,2,dtype=x.dtype)
    obj = RepulsiveEnsemble(normal,duplicates,step_size=.1)
    torch.testing.assert_close(obj.step().position,.9*duplicates)
    far = torch.tensor([[-1e5],[1e5]],dtype=x.dtype)
    assert torch.isfinite(RepulsiveEnsemble(normal,far).step().position).all()

@pytest.mark.parametrize('cls',[SVGD,SteinVariationalNewton,RepulsiveEnsemble])
def test_checkpoint_continuation(cls):
    x = torch.tensor([[-1.,.3],[.5,-.2]],dtype=torch.float64)
    obj = cls(normal,x,step_size=.03)
    obj.step(); state=obj.state_dict()
    other=cls(normal,x,step_size=.9)
    other.load_state_dict(state)
    torch.testing.assert_close(obj.step().position,other.step().position,atol=0,rtol=0)
    state['position'].zero_()
    assert other.position.abs().sum()>0

@pytest.mark.parametrize('damping',[0.,.23])
def test_svn_block_matches_transport_kl_autodiff(damping):
    x=torch.tensor([[-.7,.3],[.8,-1.2],[.2,.9]],dtype=torch.float64)
    a=torch.tensor([[1.7,.2],[.2,.9]],dtype=x.dtype)
    potential=lambda y:(y@a*y).sum(-1)/2
    ell=1.1
    def objective(coeff):
        total=x.new_zeros(())
        for point in x:
            diff=point[None]-x
            kval=torch.exp(-diff.square().sum(-1)/(2*ell**2))
            w=(kval[:,None]*coeff).sum(0)
            grad_query=-diff*kval[:,None]/ell**2
            jacobian=coeff.T@grad_query
            total+=potential(point+w)-torch.linalg.slogdet(torch.eye(2,dtype=x.dtype)+jacobian).logabsdet
        return total/len(x)
    hessian=torch.autograd.functional.hessian(objective,torch.zeros_like(x))
    method=SteinVariationalNewton(potential,x,kernel=RBF(ell),damping=damping,cg_tolerance=1e-12)
    for s in range(len(x)):
        block=hessian[s,:,s,:]+damping*torch.eye(2,dtype=x.dtype)
        v=torch.tensor([.9,-.3],dtype=x.dtype)
        torch.testing.assert_close(method.block_operator(x,s,v),block@v)
    rhs=pair_reference(x,-x@a,RBF(ell),x.new_full((3,),1/3))
    coeff=torch.stack([torch.linalg.solve(hessian[s,:,s,:]+damping*torch.eye(2,dtype=x.dtype),rhs[s]) for s in range(len(x))])
    gram=torch.exp(-((x[:,None]-x[None,:])**2).sum(-1)/(2*ell**2))
    expected=x+method.step_size*(gram@coeff)
    result=method.step()
    torch.testing.assert_close(result.position,expected)
    assert result.stats['cg_solves']==3
    assert result.stats['cg_absolute_residual'].max()<1e-10


def test_svn_damped_nonconvex_and_solver_exhaustion():
    x=torch.tensor([[1.,2.]],dtype=torch.float64)
    obj=SteinVariationalNewton(lambda y:-normal(y),x,damping=2.,step_size=.1)
    torch.testing.assert_close(obj.step().position,1.1*x)
    a=torch.diag(torch.tensor([1.,3.],dtype=x.dtype))
    obj=SteinVariationalNewton(lambda y:(y@a*y).sum(-1)/2,x,cg_max_iterations=1,cg_tolerance=1e-12)
    with pytest.raises(RuntimeError,match='converge'):obj.step()
    torch.testing.assert_close(obj.position,x,rtol=0,atol=0)


def test_svn_psd_curvature_callback():
    x=torch.tensor([[-.8,.4],[.3,1.]],dtype=torch.float64)
    exact=SteinVariationalNewton(normal,x,damping=.1)
    gn=SteinVariationalNewton(normal,x,damping=.1,curvature='gauss_newton',curvature_hvp=lambda x,v:v)
    torch.testing.assert_close(exact.step().position,gn.step().position)


def test_projected_refresh_checkpoint_and_rank_above_particles():
    x=torch.tensor([[-.8,.4,.2,.1],[.3,1.,-.4,.8]],dtype=torch.float64)
    ll=lambda y:-.5*(y[:,0]*y[:,1]-1).square()
    obj=ProjectedSVGD(ll,x,rank=3,basis_refresh=2)
    assert obj.basis.shape==(4,3)
    torch.testing.assert_close(obj.basis.T@obj.basis,torch.eye(3,dtype=x.dtype))
    obj.step(); obj.step()
    state=obj.state_dict()
    restored=ProjectedSVGD(ll,x,rank=1)
    restored.load_state_dict(state)
    torch.testing.assert_close(obj.step().position,restored.step().position,rtol=0,atol=0)
    assert obj.basis_refresh_count==2
    position=obj.position
    obj._refresh_basis()
    torch.testing.assert_close(obj.coefficients@obj.basis.T+obj.complement,position)
    torch.testing.assert_close(obj.position,position,rtol=0,atol=0)


def test_function_vjp_finite_difference_and_checkpoint():
    x=torch.tensor([[-.7,.4],[1.2,-.3]],dtype=torch.float64)
    fmap=lambda t:torch.stack([t[0].sin()+t[1].square(),t[0]*t[1]])
    obj=FunctionSpaceSVGD(fmap,normal,x)
    z=torch.stack([fmap(t) for t in x])
    phi=pair_reference(z,-z,RBF(),x.new_full((2,),.5))
    finite=torch.zeros_like(x)
    eps=1e-6
    for i in range(len(x)):
        for d in range(2):
            e=torch.zeros_like(x[i]);e[d]=eps
            finite[i,d]=((fmap(x[i]+e)-fmap(x[i]-e))*phi[i]).sum()/(2*eps)
    torch.testing.assert_close(obj.step().position,x+obj.step_size*finite,atol=1e-10,rtol=1e-10)
    other=FunctionSpaceSVGD(fmap,normal,x)
    other.load_state_dict(obj.state_dict())
    torch.testing.assert_close(obj.step().position,other.step().position,rtol=0,atol=0)


def test_repulsive_logsumexp_zero_weight_far_particle():
    from lafa.particles import kde_direction,FunctionSpaceRepulsiveEnsemble
    x=torch.tensor([[-1e4],[1e4]],dtype=torch.float64)
    w=x.new_tensor([1.,0.])
    got=kde_direction(x,torch.zeros_like(x),weights=w,block_size=1)
    torch.testing.assert_close(got,x-x[0])
    obj=FunctionSpaceRepulsiveEnsemble(lambda t:2*t,normal,x*.0001)
    z=2*x*.0001
    expected=x*.0001+.01*2*kde_direction(z,-z)
    torch.testing.assert_close(obj.step().position,expected)


def test_snapshot_does_not_retain_graph_or_mutate_state():
    x=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    obj=SVGD(normal,x)
    result=obj.step()
    assert result.position.grad_fn is None and result.gradient.grad_fn is None
    assert x.grad is None
    frozen=obj.position
    result.position.zero_(); obj.position.zero_()
    torch.testing.assert_close(obj.position,frozen)


@pytest.mark.parametrize('kwargs',[{'step_size':0},{'step_size':float('nan')},{'block_size':0},{'weights':[0,0]},{'weights':[-1,2]}])
def test_particle_configuration_validation(kwargs):
    with pytest.raises(ValueError): SVGD(normal,torch.ones(2,3),**kwargs)


def test_no_pair_coordinate_or_parameter_hessian_allocations():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    class AuditShapes(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            for t in tree_flatten(out)[0]:
                if isinstance(t,torch.Tensor):
                    assert t.ndim<=2, (func,t.shape)
                    assert t.shape!=(257,257), (func,t.shape)
            return out
    x=torch.randn(17,257,dtype=torch.float64,generator=torch.Generator().manual_seed(91))
    from lafa.diagnostics import ksd
    with AuditShapes():
        direction(x,-x,block_size=3)
        ksd(x,-x,block_size=3)
        RepulsiveEnsemble(normal,x,block_size=3).step()
        SteinVariationalNewton(normal,x,block_size=3,cg_tolerance=1e-9).step()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('method',['svgd','svn','projected','function','repulsive','function_repulsive'])
def test_all_methods_dtype_device_finiteness(method,dtype,device):
    from lafa.particles import FunctionSpaceRepulsiveEnsemble
    x=torch.tensor([[-1.,.4],[.5,-.3],[1.2,.1]],dtype=dtype,device=device)
    if method=='svgd': obj=SVGD(normal,x)
    elif method=='svn': obj=SteinVariationalNewton(normal,x)
    elif method=='projected':obj=ProjectedSVGD(lambda y:-y[:,0].square()/2,x,rank=1)
    elif method=='function':obj=FunctionSpaceSVGD(lambda t:2*t,normal,x)
    elif method=='repulsive':obj=RepulsiveEnsemble(normal,x)
    else:obj=FunctionSpaceRepulsiveEnsemble(lambda t:2*t,normal,x)
    result=obj.step()
    for tensor in (result.position,result.potential,result.gradient):
        assert tensor.dtype==dtype and tensor.device==x.device and torch.isfinite(tensor).all()


def test_projected_frozen_profile_is_not_exact_marginal_score():
    x=torch.tensor([[-1.,.3],[.2,-.9],[1.1,.6]],dtype=torch.float64)
    precision=torch.tensor([[2.,.4],[.4,1.]],dtype=x.dtype)
    ll=lambda y:-(y@precision*y).sum(-1)/2
    obj=ProjectedSVGD(ll,x,rank=1)
    v=obj.basis[:,0]
    complement_axis=torch.stack((-v[1],v[0]))
    cross=v@precision@complement_axis
    marginal_precision=v@precision@v-cross.square()/(1+complement_axis@precision@complement_axis)
    exact_marginal_score=-(1+marginal_precision)*obj.coefficients
    frozen_score=(-x@precision)@obj.basis-obj.coefficients
    assert (exact_marginal_score-frozen_score).abs().max()>.01
    expected_coeff=obj.coefficients+.01*pair_reference(obj.coefficients,frozen_score,RBF(),x.new_full((3,),1/3))
    expected=expected_coeff@obj.basis.T+obj.complement
    torch.testing.assert_close(obj.step().position,expected)


def test_nonuniform_checkpoint_bitwise_continuation():
    x=torch.tensor([[-1.,.4],[.5,-.3],[1.2,.1]],dtype=torch.float64)
    obj=SVGD(normal,x,weights=x.new_tensor([.1,.7,.2]))
    obj.step()
    other=SVGD(normal,x)
    other.load_state_dict(obj.state_dict())
    torch.testing.assert_close(obj.step().position,other.step().position,rtol=0,atol=0)


def test_float32_repulsion_translation_invariance():
    from lafa.particles import kde_direction
    x=torch.tensor([[-1.,0.],[1.,0.],[0.,1.]],dtype=torch.float32)
    for field in (direction,kde_direction):
        baseline=field(x,torch.zeros_like(x),block_size=2)
        shifted=field(x+10000,torch.zeros_like(x),block_size=2)
        torch.testing.assert_close(shifted,baseline,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('block_size',[1,2,3])
def test_mixed_scale_repulsion_preserves_near_particle_separation(block_size):
    from lafa.particles import kde_direction
    x=torch.tensor([[-1e8],[0.],[1.]],dtype=torch.float32)
    q=math.exp(-.5)
    torch.testing.assert_close(direction(x,torch.zeros_like(x),block_size=block_size),x.new_tensor([[0.],[-q/3],[q/3]]),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(kde_direction(x,torch.zeros_like(x),block_size=block_size),x.new_tensor([[0.],[-q/(1+q)],[q/(1+q)]]),rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('method',['svgd','svn','projected','function','repulsive','function_repulsive'])
def test_particle_target_graphs_respect_block_size(method):
    from lafa.particles import FunctionSpaceRepulsiveEnsemble
    calls=[]
    def target(x):
        calls.append(len(x))
        return normal(x)
    x=torch.randn(7,3,dtype=torch.float64,generator=torch.Generator().manual_seed(22))
    common=dict(block_size=2)
    if method=='svgd':obj=SVGD(target,x,**common)
    elif method=='svn':obj=SteinVariationalNewton(target,x,**common)
    elif method=='projected':obj=ProjectedSVGD(lambda t:-target(t),x,rank=1,**common)
    elif method=='function':obj=FunctionSpaceSVGD(lambda t:2*t,target,x,**common)
    elif method=='repulsive':obj=RepulsiveEnsemble(target,x,**common)
    else:obj=FunctionSpaceRepulsiveEnsemble(lambda t:2*t,target,x,**common)
    obj.step()
    assert calls and max(calls)<=2
