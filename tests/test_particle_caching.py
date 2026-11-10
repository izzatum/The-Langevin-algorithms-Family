"""Stein particle steps reuse the evaluation of the accepted candidate.

Every reuse must be bitwise invisible: a twin that calls invalidate_cache()
before each step runs the two-evaluation algorithm and is the reference.
"""
import copy
import pickle
import pytest
import torch
from lafa.particles import (SVGD, WeightSpaceSVGD, ProjectedSVGD, SteinVariationalNewton,
    FunctionSpaceSVGD, RepulsiveEnsemble, FunctionSpaceRepulsiveEnsemble)

N = 5
PARAMETER = [SVGD, SteinVariationalNewton, RepulsiveEnsemble]
FUNCTION = [FunctionSpaceSVGD, FunctionSpaceRepulsiveEnsemble]
CASES = [(SVGD,{}),(WeightSpaceSVGD,{'block_size':2}),(SteinVariationalNewton,{}),(RepulsiveEnsemble,{'block_size':3}),
         (FunctionSpaceSVGD,{}),(FunctionSpaceSVGD,{'vectorized':True,'chunk_size':2}),
         (FunctionSpaceRepulsiveEnsemble,{'block_size':2}),(FunctionSpaceRepulsiveEnsemble,{'vectorized':True})]
every_method = pytest.mark.parametrize('cls,kwargs',CASES,ids=lambda value:getattr(value,'__name__',None)
                                       or '-'.join(f'{k}={v}' for k,v in value.items()) or 'default')


class Counted:
    """Deterministic convex target. Counts value_and_grad batches apart from direct (HVP) calls."""
    def __init__(self,scale=1.):
        self.scale,self.batches,self.fail_from = scale,[],None

    def __call__(self,x):
        return self.scale*(x.square().sum(-1)/2+torch.logsumexp(x,-1))

    def value_and_grad(self,x):
        if self.fail_from is not None and len(self.batches) >= self.fail_from: raise RuntimeError('target failure')
        self.batches.append(len(x))
        with torch.enable_grad():
            leaf = x.detach().clone().requires_grad_(True)
            value = self(leaf)
            gradient = torch.autograd.grad(value.sum(),leaf)[0]
        return value.detach(),gradient


class CountedMap:
    """Deterministic vmap-compatible function map counting its Python-level calls."""
    def __init__(self):
        self.calls,self.fail = 0,False

    def __call__(self,t):
        if self.fail: raise RuntimeError('function map failure')
        self.calls += 1
        return torch.stack([t[0].sin()+t[1].square(),t[0]*t[1],t.sum().cos()])


def start(seed=5):
    return torch.randn(N,3,dtype=torch.float64,generator=torch.Generator().manual_seed(seed))


def make(cls,*,x=None,scale=1.,**kwargs):
    """(method, counted target, counted function map or None) on one fixed problem."""
    x,target = start() if x is None else x,Counted(scale)
    if issubclass(cls,FunctionSpaceSVGD):
        function_map = CountedMap()
        return cls(function_map,target,x,step_size=.05,**kwargs),target,function_map
    if cls is SteinVariationalNewton: kwargs = dict(damping=.1,cg_tolerance=1e-9)|kwargs
    return cls(target,x,step_size=.05,**kwargs),target,None


def blocks(kwargs):
    """Target batches per evaluation of the ensemble."""
    return -(-N//kwargs.get('block_size',128))


def passes(kwargs):
    """function_map calls per sweep: one per particle, or one per vmapped chunk."""
    return -(-N//(kwargs.get('chunk_size') or N)) if kwargs.get('vectorized') else N


def assert_tree(a,b):
    assert type(a) is type(b)
    if isinstance(a,torch.Tensor): assert a.dtype == b.dtype and torch.equal(a,b)
    elif isinstance(a,dict):
        assert list(a) == list(b)
        for key in a: assert_tree(a[key],b[key])
    else: assert a == b


def assert_same(result,expected):
    for name in ('position','potential','gradient'):
        assert torch.equal(getattr(result,name),getattr(expected,name)), name
    assert result.accepted is None and expected.accepted is None
    assert_tree(result.stats,expected.stats)


def cost(obj,target,function_map):
    """(target batches, function_map calls) spent by one step."""
    before = len(target.batches),function_map.calls if function_map else 0
    obj.step()
    return len(target.batches)-before[0],(function_map.calls if function_map else 0)-before[1]


@pytest.mark.parametrize('cls',PARAMETER)
@pytest.mark.parametrize('block_size',[128,2])
def test_ten_steps_cost_eleven_target_batches_per_block(cls,block_size):
    obj,target,_ = make(cls,block_size=block_size)
    assert target.batches == []
    per_step = [cost(obj,target,None)[0] for _ in range(10)]
    count = blocks({'block_size':block_size})
    assert per_step == [2*count]+[count]*9  # Without reuse: 2*count every step, 20 per block.
    assert len(target.batches) == 11*count and max(target.batches) <= block_size


@pytest.mark.parametrize('cls',FUNCTION)
@pytest.mark.parametrize('kwargs',[{},{'block_size':2},{'vectorized':True},{'vectorized':True,'chunk_size':2}])
def test_function_step_costs_one_output_pass_and_two_vjp_sweeps(cls,kwargs):
    obj,target,function_map = make(cls,**kwargs)
    assert function_map.calls == passes(kwargs) and target.batches == []
    spent = [cost(obj,target,function_map) for _ in range(10)]
    # The start is evaluated once (outputs and VJP); afterwards three sweeps per
    # step, i.e. three function_map calls per particle in loop mode. Without
    # reuse every step costs six sweeps and three function-potential evaluations.
    assert [calls for _,calls in spent] == [5*passes(kwargs)]+[3*passes(kwargs)]*9
    assert [batches for batches,_ in spent] == [2*blocks(kwargs)]+[blocks(kwargs)]*9


@every_method
def test_reuse_is_bitwise_equal_to_invalidating_every_step(cls,kwargs):
    reused,target,_ = make(cls,**kwargs)
    twin,twin_target,_ = make(cls,**kwargs)
    for _ in range(6):
        twin.invalidate_cache()
        assert_same(reused.step(),twin.step())
    assert torch.equal(reused.position,twin.position)
    assert_tree(reused.state_dict(),twin.state_dict())
    assert len(target.batches) == 7*blocks(kwargs) and len(twin_target.batches) == 12*blocks(kwargs)


@every_method
def test_nonfinite_candidate_leaves_position_and_retained_evaluation_usable(cls,kwargs):
    obj,target,function_map = make(cls,**kwargs)
    twin,_,_ = make(cls,**kwargs)
    for _ in range(2): assert_same(obj.step(),twin.step())
    position,state = obj.position,obj.state_dict()
    obj.step_size = float('inf')
    with pytest.raises(ValueError,match='finite'): obj.step()
    obj.step_size = state['step_size']
    assert torch.equal(obj.position,position)
    assert_tree(obj.state_dict(),state)
    before = len(target.batches),function_map.calls if function_map else 0
    assert_same(obj.step(),twin.step())
    # The failed step did not spoil the evaluation of the unchanged position.
    assert len(target.batches)-before[0] == blocks(kwargs)
    if function_map: assert function_map.calls-before[1] == 3*passes(kwargs)


@every_method
@pytest.mark.parametrize('warm',[0,2])
def test_target_failure_at_the_candidate_then_retry_matches_unfailed_twin(cls,kwargs,warm):
    obj,target,function_map = make(cls,**kwargs)
    twin,_,_ = make(cls,**kwargs)
    for _ in range(warm): assert_same(obj.step(),twin.step())
    state = obj.state_dict()
    # A cold step first evaluates the start; the candidate evaluation then raises.
    target.fail_from = len(target.batches)+(0 if warm else blocks(kwargs))
    with pytest.raises(RuntimeError,match='target failure'): obj.step()
    target.fail_from = None
    assert_tree(obj.state_dict(),state)
    before = len(target.batches)
    assert_same(obj.step(),twin.step())
    assert len(target.batches)-before == blocks(kwargs)
    assert_same(obj.step(),twin.step())


@pytest.mark.parametrize('cls',FUNCTION)
@pytest.mark.parametrize('kwargs',[{},{'vectorized':True,'chunk_size':2}])
def test_function_map_failure_then_retry_matches_unfailed_twin(cls,kwargs):
    obj,target,function_map = make(cls,**kwargs)
    twin,_,_ = make(cls,**kwargs)
    assert_same(obj.step(),twin.step())
    state = obj.state_dict()
    function_map.fail = True
    with pytest.raises(RuntimeError,match='function map failure'): obj.step()
    function_map.fail = False
    assert_tree(obj.state_dict(),state)
    assert cost(obj,target,function_map) == (blocks(kwargs),3*passes(kwargs))
    twin.step()
    assert_same(obj.step(),twin.step())


def test_newton_solver_failure_keeps_the_evaluated_start():
    obj,target,_ = make(SteinVariationalNewton)
    twin,_,_ = make(SteinVariationalNewton)
    obj.cg_max_iterations = 1
    with pytest.raises(RuntimeError,match='converge'): obj.step()
    assert target.batches == [N] and obj.iteration == 0 and torch.equal(obj.position,start())
    obj.cg_max_iterations = 100
    assert_same(obj.step(),twin.step())
    assert target.batches == [N,N]  # The start was not evaluated a second time.


@every_method
def test_load_state_dict_drops_the_retained_evaluation(cls,kwargs):
    obj,target,function_map = make(cls,**kwargs)
    obj.step()
    checkpoint = obj.state_dict()
    obj.step(); obj.step()
    assert obj._cache is not None
    obj.load_state_dict(checkpoint)  # Rewind a warm object.
    assert obj._cache is None
    fresh,_,_ = make(cls,**kwargs)
    fresh.load_state_dict(checkpoint)
    before = len(target.batches)
    assert_same(obj.step(),fresh.step())
    assert len(target.batches)-before == 2*blocks(kwargs)  # The restored position was evaluated.
    assert cost(obj,target,function_map)[0] == blocks(kwargs)
    # Reloading identical coordinates is still a restore: a target changed
    # meanwhile must be evaluated again, not answered from before.
    target.scale = 3.
    obj.load_state_dict(obj.state_dict())
    changed,_,_ = make(cls,scale=3.,**kwargs)
    changed.load_state_dict(obj.state_dict())
    assert_same(obj.step(),changed.step())


@pytest.mark.parametrize('cls',[SVGD,FunctionSpaceSVGD])
def test_state_dict_is_unchanged_and_base_format_checkpoints_load(cls):
    obj,_,_ = make(cls)
    cold = obj.state_dict()
    for _ in range(2): obj.step()
    state = obj.state_dict()
    assert list(state) == list(cold) == ['version','method','position','iteration','step_size','block_size',
                                         'weights','kernel','extra','reattach']
    extra = {'vectorized':False,'chunk_size':None} if cls is FunctionSpaceSVGD else {}
    assert state['extra'] == extra and state['kernel'] == {'kind':'rbf','length_scale':1.}
    # The dictionary a base-version object wrote for this position.
    legacy = {'version':1,'method':cls.__name__,'position':obj.position,'iteration':2,'step_size':.05,
              'block_size':128,'weights':torch.full((N,),1/N,dtype=torch.float64),
              'kernel':{'kind':'rbf','length_scale':1.},'extra':extra,
              'reattach':'target/function callbacks are not serialized'}
    assert_tree(state,legacy)
    restored,_,_ = make(cls)
    restored.load_state_dict(legacy)
    assert_tree(restored.state_dict(),legacy)
    assert_same(restored.step(),obj.step())


@every_method
def test_invalidate_cache_after_changing_target_data_in_place(cls,kwargs):
    obj,target,_ = make(cls,**kwargs)
    obj.step(); obj.step()
    target.scale = 3.
    obj.invalidate_cache()
    reference,_,_ = make(cls,scale=3.,**kwargs)
    reference.load_state_dict(obj.state_dict())
    for _ in range(2): assert_same(obj.step(),reference.step())


@every_method
def test_replaced_target_object_is_not_served_the_old_evaluation(cls,kwargs):
    obj,target,function_map = make(cls,**kwargs)
    obj.step(); obj.step()
    replacement = Counted(3.)
    if function_map is None: obj.potential = replacement
    else: obj.function_potential = replacement
    reference,_,_ = make(cls,scale=3.,**kwargs)
    reference.load_state_dict(obj.state_dict())
    before = len(target.batches)
    assert_same(obj.step(),reference.step())
    assert len(replacement.batches) == 2*blocks(kwargs)  # The position was evaluated under the new target.
    assert_same(obj.step(),reference.step())
    assert len(replacement.batches) == 3*blocks(kwargs) and len(target.batches) == before


@pytest.mark.parametrize('cls',FUNCTION)
def test_replaced_function_map_is_not_served_the_old_outputs(cls):
    obj,target,_ = make(cls)
    obj.step()
    obj.function_map = lambda t:torch.stack([t[0]*t[1],t[1].cos(),t.sum()])
    reference,_,_ = make(cls)
    reference.function_map = obj.function_map
    reference.load_state_dict(obj.state_dict())
    before = len(target.batches)
    for _ in range(2): assert_same(obj.step(),reference.step())
    assert len(target.batches)-before == 3  # Evaluated once under the new map, then retained.


@every_method
def test_changed_batching_is_not_served_the_old_evaluation(cls,kwargs):
    obj,target,function_map = make(cls,**kwargs)
    obj.step()
    obj.block_size = 1
    reference,_,_ = make(cls,**kwargs)
    reference.load_state_dict(obj.state_dict())
    before = len(target.batches)
    assert_same(obj.step(),reference.step())
    assert target.batches[before:] == [1]*2*N  # Evaluated again, in the new batches.
    assert_same(obj.step(),reference.step())
    assert target.batches[before:] == [1]*3*N  # Retained again from then on.
    if function_map:
        obj.vectorized,obj.chunk_size = not obj.vectorized,3
        reference.load_state_dict(obj.state_dict())
        calls = function_map.calls
        assert_same(obj.step(),reference.step())
        assert function_map.calls-calls == 5*passes({'vectorized':obj.vectorized,'chunk_size':3})


def test_inference_mode_change_is_not_served_the_other_mode_evaluation():
    # Autograd targets yield no gradient inside inference mode, so an evaluation
    # made outside it must not be reused inside, and the reverse.
    obj = SVGD(lambda x:x.square().sum(-1)/2,start(),step_size=.05)
    twin = SVGD(lambda x:x.square().sum(-1)/2,start(),step_size=.05)
    assert_same(obj.step(),twin.step())
    for inside in (True,True,False,False):
        twin.invalidate_cache()
        with torch.inference_mode(inside): assert_same(obj.step(),twin.step())
    assert torch.equal(obj.position,twin.position)


@every_method
def test_in_place_position_edit_is_not_served_the_old_evaluation(cls,kwargs):
    obj,target,_ = make(cls,**kwargs)
    obj.step()
    obj._position.mul_(.5)  # Same tensor object, different particles.
    reference,_,_ = make(cls,**kwargs)
    reference.load_state_dict(obj.state_dict())
    before = len(target.batches)
    assert_same(obj.step(),reference.step())
    assert len(target.batches)-before == 2*blocks(kwargs)
    assert_same(obj.step(),reference.step())
    assert len(target.batches)-before == 3*blocks(kwargs)


@every_method
def test_step_results_do_not_alias_the_retained_evaluation(cls,kwargs):
    obj,_,_ = make(cls,**kwargs)
    twin,_,_ = make(cls,**kwargs)
    for _ in range(3):
        result,expected = obj.step(),twin.step()
        assert_same(result,expected)
        for tensor in (result.position,result.potential,result.gradient): tensor.fill_(float('nan'))
        twin.invalidate_cache()
    assert torch.equal(obj.position,twin.position)


def pickled(obj):
    # Round trip of an object this test built itself; no external data is unpickled.
    return pickle.loads(pickle.dumps(obj))


@pytest.mark.parametrize('cls',[SVGD,FunctionSpaceSVGD])
@pytest.mark.parametrize('duplicate',[copy.deepcopy,pickled])
def test_copies_carry_no_retained_evaluation_and_step_like_the_original(cls,duplicate):
    obj,target,function_map = make(cls)
    obj.step()
    clone = duplicate(obj)
    clone_target = clone.function_potential if function_map else clone.potential
    clone_map = clone.function_map if function_map else None
    assert obj._cache is not None and clone._cache is None and set(vars(clone)) == set(vars(obj))-{'_cache'}
    # The copy evaluates its own position first; the original still reuses.
    assert cost(obj,target,function_map) == (1,3*N if function_map else 0)
    assert cost(clone,clone_target,clone_map) == (2,5*N if function_map else 0)
    assert torch.equal(clone.position,obj.position)
    assert_same(clone.step(),obj.step())
    assert_tree(clone.state_dict(),obj.state_dict())


@pytest.mark.parametrize('cls',FUNCTION)
def test_function_direction_of_another_tensor_is_recomputed(cls):
    obj,target,function_map = make(cls)
    obj.step()
    retained,_ = obj._direction(obj._position,None)
    recomputed,_ = obj._direction(obj.position,None)  # Equal coordinates, another tensor.
    assert torch.equal(retained,recomputed)
    other = start(seed=9)
    fresh,_,_ = make(cls,x=other)
    assert torch.equal(obj._direction(other,None)[0],fresh._direction(other,None)[0])
    assert not torch.equal(obj._direction(other,None)[0],retained)
    assert cost(obj,target,function_map) == (1,3*N)  # The current position is still retained.


@pytest.mark.parametrize('refresh',[None,2])
def test_projected_svgd_matches_its_invalidating_twin_and_retains_nothing(refresh):
    def likelihood(log):
        def log_likelihood(y):
            log.append(len(y))
            return -.5*(y[:,0]*y[:,1]-1).square()-.1*(y[:,2]-y[:,0]).square()
        return log_likelihood
    logs = [],[]
    obj = ProjectedSVGD(likelihood(logs[0]),start(),rank=2,basis_refresh=refresh,step_size=.05,block_size=2)
    twin = ProjectedSVGD(likelihood(logs[1]),start(),rank=2,basis_refresh=refresh,step_size=.05,block_size=2)
    for _ in range(5):
        twin.invalidate_cache()
        assert_same(obj.step(),twin.step())
    assert_tree(obj.state_dict(),twin.state_dict())
    # It evaluates the likelihood gradient and then the full potential, two
    # different functions, so every step still costs both (plus basis refreshes).
    assert logs[0] == logs[1] and len(logs[0]) == 3*(1+2*5+(2 if refresh else 0))
    assert obj._cache is None
    # Iteration 5 is not a refresh step, so the failed attempt changes nothing at all.
    state = obj.state_dict()
    obj.step_size = float('inf')
    with pytest.raises(ValueError,match='finite'): obj.step()
    obj.step_size = .05
    assert_tree(obj.state_dict(),state)
    twin.invalidate_cache()
    assert_same(obj.step(),twin.step())
