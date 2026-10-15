"""Execution controls preserve kernels, owned outputs, and transactional adapters."""
import copy
import pytest
import torch
from lafa import samplers, optim

NAMES = ['ULA', 'MALA', 'LipULA', 'LipMALA', 'BAOAB', 'KineticLangevin']


def target(x):
    return x.square().sum(-1)/2


def make(name, **kwargs):
    return getattr(samplers, name)(target, torch.ones(3, 2, dtype=torch.float64),
                                  step_size=.05, generator=torch.Generator().manual_seed(23), **kwargs)


def assert_nested(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize('name', NAMES)
def test_result_modes_have_identical_state_rng_history_and_checkpoint(name):
    full, selected, empty, advanced = [make(name) for _ in range(4)]
    saved = selected.state_dict()
    for _ in range(5):
        result = full.step()
        subset = selected.step(fields=('position', 'accepted', 'stats'))
        assert empty.step(fields=()) is None
        assert torch.equal(result.position, subset['position'])
        assert_nested(result.stats, subset['stats'])
        assert_nested(full.state_dict(), selected.state_dict())
        assert_nested(full.state_dict(), empty.state_dict())
    assert advanced.advance(5) is None
    assert_nested(full.state_dict(), advanced.state_dict())
    selected.load_state_dict(saved)
    selected.advance(5)
    assert_nested(full.state_dict(), selected.state_dict())
    retained = copy.deepcopy(subset)
    selected.advance(2)
    assert_nested(subset, retained)


@pytest.mark.parametrize('fields', [('unknown',), 'position', ('position', 'position'), [None], 1])
def test_invalid_fields_fail_before_state_or_rng_change(fields):
    sampler = make('ULA')
    before = sampler.state_dict()
    with pytest.raises((TypeError, ValueError)):
        sampler.step(fields=fields)
    assert_nested(sampler.state_dict(), before)


@pytest.mark.parametrize('steps', [-1, .5, True])
def test_invalid_advance_rejected_before_transition(steps):
    sampler = make('ULA')
    before = sampler.state_dict()
    with pytest.raises((TypeError, ValueError)):
        sampler.advance(steps)
    assert_nested(sampler.state_dict(), before)
    assert sampler.advance(0) is None


@pytest.mark.parametrize('name', ['MALA', 'LipMALA'])
def test_invalid_mala_proposal_in_no_result_mode_rejects(name):
    sampler = getattr(samplers, name)(target, torch.zeros(2, dtype=torch.float32),
        step_size=1e30, temperature=1e30, generator=torch.Generator().manual_seed(1),
        **({'max_step_size':1e30} if name == 'LipMALA' else {}))
    assert sampler.step(fields=()) is None
    assert torch.equal(sampler.position, torch.zeros(2))
    assert sampler.iteration == 1


@pytest.mark.parametrize('name', ['ULA', 'MALA', 'LipULA', 'LipMALA', 'BAOAB'])
@pytest.mark.parametrize('after_first', [False, True])
def test_optimizer_exception_restores_parameters_gradients_sampler_and_rng(name, after_first):
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = getattr(optim, name)([parameter], lr=.05, generator=torch.Generator().manual_seed(19))
    def closure():
        optimizer.zero_grad()
        value = parameter.square().sum()/2
        value.backward()
        return value
    if after_first:
        optimizer.step(closure)
    parameter.grad = torch.full_like(parameter, 9.)
    before = optimizer.state_dict()
    previous_result = optimizer.last_result
    calls = 0
    def broken():
        nonlocal calls
        calls += 1
        value = closure()
        # ULA fails after closure; adapters fail during proposal evaluation.
        if name == 'ULA':
            parameter.grad.fill_(float('inf'))
        elif calls == 2:
            raise RuntimeError('proposal closure failed')
        return value
    with pytest.raises((RuntimeError, ValueError)):
        optimizer.step(broken)
    assert_nested(optimizer.state_dict(), before)
    assert torch.equal(parameter.grad, torch.full_like(parameter, 9.))
    assert optimizer.last_result is previous_result
    # Recovery consumes exactly the same randomness as resuming the checkpoint.
    optimizer.step(closure)
    recovered = parameter.detach().clone()
    optimizer.load_state_dict(before)
    optimizer.step(closure)
    assert torch.equal(parameter, recovered)


def test_optimizer_scales_reused_but_group_lr_external_parameters_and_closure_are_fresh():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = optim.ULA([p], lr=.1, generator=torch.Generator().manual_seed(11))
    scales = optimizer._scales()
    assert scales is optimizer._scales()
    optimizer.param_groups[0]['lr'] = .3
    updated = optimizer._scales()
    assert updated is not scales
    assert torch.equal(updated, torch.full_like(p, .3))
    with torch.no_grad():
        p.fill_(5.)
    rng = torch.Generator().set_state(optimizer.generator.get_state())
    noise = torch.randn(p.shape, dtype=p.dtype, generator=rng)
    def closure():
        optimizer.zero_grad(); value = p.square().sum()/2; value.backward(); return value
    optimizer.step(closure)
    torch.testing.assert_close(p, torch.full_like(p, 3.5)+(2*.3)**.5*noise)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('name', ['ULA', 'MALA', 'BAOAB'])
def test_compiled_target_matches_eager_sampling(dtype, name):
    from lafa.execution import compile_potential
    initial = torch.ones(3, 2, dtype=dtype)
    # aot_eager exercises the captured functional derivative without expensive codegen.
    compiled = compile_potential(target, backend='aot_eager')
    eager = getattr(samplers, name)(target, initial, generator=torch.Generator().manual_seed(4))
    fast = getattr(samplers, name)(compiled, initial, generator=torch.Generator().manual_seed(4))
    for _ in range(3):
        torch.testing.assert_close(fast.step(fields=('position',))['position'], eager.step().position)
    assert torch.equal(eager.generator.get_state(), fast.generator.get_state())


def test_compile_rejects_score_and_custom_evaluation_protocols():
    from lafa.execution import compile_potential
    from lafa.targets import from_score
    with pytest.raises(TypeError, match='pure tensor callable'):
        compile_potential(from_score(lambda x: -x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA hardware required')
def test_selective_cuda_owned_output():
    sampler = samplers.ULA(target, torch.ones(2, 3, device='cuda'), generator=torch.Generator(device='cuda').manual_seed(1))
    result = sampler.step(fields=('position',))
    retained = result['position'].clone()
    sampler.advance(2)
    assert torch.equal(result['position'], retained)


@pytest.mark.parametrize('name', NAMES)
def test_selected_outputs_owned_and_complete_without_snapshot_allocations(name, monkeypatch):
    sampler = make(name)
    full = sampler.step()
    full_position = full.position.clone()
    selected = sampler.step(fields=('position','potential','gradient','accepted','stats'))
    assert set(selected) == {'position','potential','gradient','accepted','stats'}
    retained = copy.deepcopy(selected)
    for value in (selected['position'], selected['potential'], selected['gradient']):
        value.add_(10.)
    # User mutation cannot modify the current numerical state.
    assert not torch.equal(selected['position'], sampler.position)
    saved = sampler.state_dict()
    def forbidden(*args, **kwargs):
        raise AssertionError('advance must not allocate snapshots')
    monkeypatch.setattr(sampler, '_snapshot', forbidden)
    sampler.advance(3)
    assert torch.equal(full.position, full_position)
    assert not torch.equal(saved['state']['_position'], sampler.position)
    assert retained['position'].shape == sampler.position.shape


@pytest.mark.parametrize('name', ['LipULA','LipMALA'])
def test_no_result_adaptation_freezes_at_production_boundary(name):
    sampler = make(name, adapt_steps=2)
    sampler.advance(2)
    frozen = sampler.step_size.clone()
    sampler.advance(8)
    assert torch.equal(sampler.step_size, frozen)
    assert not sampler.step(fields=('stats',))['stats']['adapting']


@pytest.mark.parametrize('name', ['ULA','LipULA','BAOAB','KineticLangevin'])
def test_no_result_mode_keeps_nonfinite_checks(name):
    sampler = make(name)
    before = sampler.position
    sampler._evaluate = lambda x: (torch.full_like(x[...,0], float('nan')), torch.full_like(x,float('nan')))
    with pytest.raises(ValueError, match='nonfinite'):
        sampler.step(fields=())
    assert torch.equal(before, sampler.position)
    assert sampler.iteration == 0


@pytest.mark.parametrize('name', ['MALA','LipULA','BAOAB'])
def test_optimizer_failure_after_numerical_commit_restores_history(name, monkeypatch):
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = getattr(optim,name)([p], lr=.05, generator=torch.Generator().manual_seed(15))
    def closure():
        optimizer.zero_grad(); value=p.square().sum()/2; value.backward(); return value
    optimizer.step(closure)
    before = optimizer.state_dict()
    original = optimizer._assign
    def fail_assignment(x, gradient=None):
        if gradient is not None:
            raise RuntimeError('assignment failed after sampler commit')
        return original(x, gradient)
    monkeypatch.setattr(optimizer, '_assign', fail_assignment)
    with pytest.raises(RuntimeError, match='after sampler commit'):
        optimizer.step(closure)
    assert_nested(optimizer.state_dict(), before)


@pytest.mark.parametrize('name', ['MALA','LipULA','BAOAB'])
def test_optimizer_pending_checkpoint_survives_failed_first_step(name):
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = getattr(optim,name)([p], lr=.05, generator=torch.Generator().manual_seed(15))
    def closure():
        optimizer.zero_grad(); value=p.square().sum()/2; value.backward(); return value
    optimizer.step(closure)
    before = optimizer.state_dict()
    optimizer.load_state_dict(before)
    calls=0
    def broken():
        nonlocal calls
        calls += 1
        value = closure()
        if calls == 2:
            raise RuntimeError('failed after checkpoint load')
        return value
    with pytest.raises(RuntimeError, match='checkpoint load'):
        optimizer.step(broken)
    assert optimizer._sampler is None
    assert optimizer._pending_sampler_state is not None
    assert_nested(optimizer.state_dict(), before)


def test_compilation_failure_is_explicit_without_silent_fallback():
    from lafa.execution import compile_potential
    def unsupported(x):
        # Python scalar conversion is deliberately excluded from fullgraph.
        return x.square().sum(-1) + x.sum().item()
    sampler_initial=torch.ones(2,dtype=torch.float64)
    with pytest.raises(RuntimeError, match='compiled potential/gradient evaluation failed') as error:
        samplers.ULA(compile_potential(unsupported,backend='aot_eager'), sampler_initial)
    assert error.value.__cause__ is not None


def test_compiled_constant_target_and_no_grad():
    from lafa.execution import compile_potential
    with torch.no_grad():
        sampler=samplers.ULA(compile_potential(lambda x: torch.ones_like(x[...,0]),backend='aot_eager'),torch.ones(2))
        assert sampler.step(fields=('gradient',))['gradient'].eq(0).all()


@pytest.mark.parametrize('name', ['MALA','BAOAB'])
def test_loaded_optimizer_respects_external_parameter_and_lr_before_first_step(name):
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
    optimizer=getattr(optim,name)([p],lr=.05,generator=torch.Generator().manual_seed(2))
    def closure():
        optimizer.zero_grad(); value=p.square().sum()/2; value.backward(); return value
    optimizer.step(closure)
    checkpoint=optimizer.state_dict()
    optimizer.load_state_dict(checkpoint)
    with torch.no_grad():
        p.fill_(3.)
    optimizer.param_groups[0]['lr']=.2
    # Materialize numerical history before making the same external edits.
    q=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
    reference=getattr(optim,name)([q],lr=.05,generator=torch.Generator().manual_seed(2))
    def other_closure():
        reference.zero_grad(); value=q.square().sum()/2; value.backward(); return value
    reference.step(other_closure)
    with torch.no_grad():
        q.fill_(3.)
    reference.param_groups[0]['lr']=.2
    reference.step(other_closure)
    optimizer.step(closure)
    assert torch.equal(p,q)
    assert torch.equal(optimizer._sampler.step_size,torch.tensor(.2,dtype=p.dtype))


@pytest.mark.parametrize('name', NAMES)
def test_native_target_exception_has_identical_rng_consumption_and_checkpoint_retry(name):
    full, suppressed = make(name), make(name)
    checkpoint = full.state_dict()
    def failing(x):
        raise RuntimeError('target failed')
    for sampler, fields in [(full,None),(suppressed,())]:
        sampler.potential_fn = failing
        with pytest.raises(RuntimeError, match='target failed'):
            sampler.step(fields=fields)
    assert_nested(full.state_dict(), suppressed.state_dict())
    assert full.iteration == 0
    assert torch.equal(full.position, checkpoint['state']['_position'])
    assert not torch.equal(full.generator.get_state(), checkpoint['generator_state'])
    full.potential_fn = target
    full.load_state_dict(checkpoint)
    assert torch.equal(full.step().position, make(name).step().position)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_compiled_nonlinear_value_and_gradient_at_nonzero_position(dtype):
    from lafa.execution import compile_potential
    from lafa.targets import value_and_grad
    matrix=torch.tensor([[.3,.7],[-.2,.5]],dtype=dtype)
    def nonlinear(x):
        return x.square().sum(-1)/2 + .05*(x@matrix).sin().square().sum(-1)
    position=torch.tensor([[.7,-.2],[1.1,.6]],dtype=dtype)
    expected=value_and_grad(nonlinear,position)
    actual=value_and_grad(compile_potential(nonlinear,backend='aot_eager'),position)
    for left,right in zip(actual,expected):
        torch.testing.assert_close(left,right)
