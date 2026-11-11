"""Optimizer adapters: group forms, interrupted-step rollback, Lip lr changes and cached gradient reuse."""
import types
import pytest
import torch
import lafa

DT = torch.float64
ADAPTERS = ['ULA', 'MALA', 'LipULA', 'LipMALA', 'BAOAB']
CLOSURE_ADAPTERS = ['MALA', 'LipULA', 'LipMALA', 'BAOAB']
LIP = ['LipULA', 'LipMALA']
# Two lr groups where the kernel allows them. With seed 7 MALA and LipMALA both accept and reject within ten steps.
CASES = {'ULA': dict(lrs=(.05, .1)), 'MALA': dict(lrs=(.2, .4)), 'LipULA': dict(lrs=(.05, .1)),
         'LipMALA': dict(lrs=(.2, .4)), 'BAOAB': dict(lr=.1, damping=2.)}


def make(name, *, seed=7, lr=.05, lrs=None, **kwargs):
    """Optimizer over two tensors (4+3 coordinates) with a counting deterministic potential closure."""
    a = torch.nn.Parameter(torch.tensor([[1., -.5], [.25, 2.]], dtype=DT))
    b = torch.nn.Parameter(torch.tensor([.5, -1., 1.5], dtype=DT))
    params = [a, b] if lrs is None else [{'params': [a], 'lr': lrs[0]}, {'params': [b], 'lr': lrs[1]}]
    optimizer = getattr(lafa, name)(params, lr=lr, generator=torch.Generator().manual_seed(seed), **kwargs)
    run = types.SimpleNamespace(optimizer=optimizer, a=a, b=b, calls=0, shift=0., hook=None)

    def closure():
        run.calls += 1
        optimizer.zero_grad()
        value = (a.square().sum()+a.pow(4).sum()/4+(b-run.shift).square().sum())/2
        value.backward()
        if run.hook is not None:
            run.hook()
        return value
    run.closure = closure
    run.flat = lambda: torch.cat([a.detach().reshape(-1), b.detach().reshape(-1)])
    run.grads = lambda: None if a.grad is None else torch.cat([a.grad.reshape(-1), b.grad.reshape(-1)])
    return run


def step(run, *, fresh=False):
    """One step; fresh passes a new closure object, as a loop defining its closure per batch does."""
    return run.optimizer.step((lambda: run.closure()) if fresh else run.closure)


def trajectory(run, steps, *, fresh=False):
    """Owned (parameters, gradients, returned value, step size used) after each step."""
    out = []
    for _ in range(steps):
        value = step(run, fresh=fresh)
        result = run.optimizer.last_result
        out.append((run.flat(), run.grads().clone(), value.detach().clone(),
                    None if result is None else result.stats['step_size'].clone()))
    return out


def resumed(name, saved):
    """Fresh optimizer with other values, seed and lr, restored from saved."""
    grouped = len(saved['param_groups']) == 2
    other = make(name, seed=999, lr=.3, lrs=(.3, .3) if grouped else None)
    with torch.no_grad():
        other.a.zero_(); other.b.zero_()
    other.optimizer.load_state_dict(saved)
    return other


def assert_nested(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested(left, right)
    else:
        assert actual == expected


def assert_same(run, twin):
    assert torch.equal(run.flat(), twin.flat())
    assert_nested(run.grads(), twin.grads())
    assert torch.equal(run.optimizer.generator.get_state(), twin.optimizer.generator.get_state())


def close(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), rtol=1e-12, atol=0)


# Parameter-group forms.

@pytest.mark.parametrize('name', ADAPTERS)
def test_group_holding_one_bare_tensor_matches_torch_optim(name):
    p, q, r = (torch.nn.Parameter(torch.tensor([[1., -2.], [.5, 3.], [0., 1.]], dtype=DT)) for _ in range(3))
    reference = torch.optim.SGD([{'params': q}], lr=.05)
    optimizer = getattr(lafa, name)([{'params': p}], lr=.05, generator=torch.Generator().manual_seed(4))
    listed = getattr(lafa, name)([{'params': [r]}], lr=.05, generator=torch.Generator().manual_seed(4))
    for candidate, tensor in [(reference, q), (optimizer, p)]:
        (group,) = candidate.param_groups
        assert isinstance(group['params'], list) and len(group['params']) == 1 and group['params'][0] is tensor
        assert group['lr'] == .05
    for candidate, tensor in [(optimizer, p), (listed, r)]:
        def closure():
            candidate.zero_grad(); value = tensor.square().sum()/2; value.backward(); return value
        assert candidate.step(closure).ndim == 0
    # The whole tensor is one parameter, so the transition equals the list form.
    assert torch.equal(p, r) and not torch.equal(p, q)


@pytest.mark.parametrize('name', ADAPTERS)
def test_bare_tensor_params_raise_type_error_like_torch_optim(name):
    p = torch.nn.Parameter(torch.ones(3, 2, dtype=DT))
    for build in (lambda: torch.optim.SGD(p, lr=.05), lambda: getattr(lafa, name)(p, lr=.05)):
        with pytest.raises(TypeError, match='iterable of Tensors or dicts'):
            build()


# Rollback on BaseException.

@pytest.mark.parametrize('name', ADAPTERS)
@pytest.mark.parametrize('warm', [0, 3])
def test_interrupted_step_is_rolled_back_and_retry_matches_uninterrupted_twin(name, warm):
    run, twin = make(name, **CASES[name]), make(name, **CASES[name])
    for _ in range(warm):
        assert torch.equal(step(run), step(twin))
    start, grads = run.flat(), run.grads()
    grads = None if grads is None else grads.clone()
    before, result, sampler = run.optimizer.state_dict(), run.optimizer.last_result, run.optimizer._sampler

    def interrupt():
        # ULA evaluates its closure at the current state; the others are interrupted at the proposal.
        if name == 'ULA' or not torch.equal(run.flat(), start):
            raise KeyboardInterrupt
    run.hook = interrupt
    with pytest.raises(KeyboardInterrupt):
        step(run)
    run.hook = None
    assert torch.equal(run.flat(), start)
    assert_nested(run.grads(), grads)
    assert torch.equal(run.optimizer.generator.get_state(), before['lafa']['generator_state'])
    # The state dict also carries the sampler's numerical history.
    assert_nested(run.optimizer.state_dict(), before)
    assert run.optimizer.last_result is result and run.optimizer._sampler is sampler
    for _ in range(3):
        assert torch.equal(step(run), step(twin))
        assert_same(run, twin)
    assert_nested(run.optimizer.state_dict(), twin.optimizer.state_dict())


# LipULA / LipMALA lr changes.

@pytest.mark.parametrize('name', LIP)
def test_lip_single_group_lr_change_sets_the_step(name):
    run = make(name, lr=.05, adapt_steps=0)
    step(run)
    assert run.optimizer.last_result.stats['step_size'] == .05
    run.optimizer.param_groups[0]['lr'] = .005
    step(run)
    close(run.optimizer.last_result.stats['step_size'], .005)
    close(run.optimizer._sampler.step_size, .005)


@pytest.mark.parametrize('name', LIP)
def test_lip_adapted_step_is_multiplied_by_the_lr_ratio_and_alpha_is_kept(name):
    run = make(name, lr=.05)
    for _ in range(4):
        step(run)
    sampler = run.optimizer._sampler
    adapted, alpha = sampler.step_size.clone(), sampler._alpha.clone()
    assert adapted != .05 and torch.isfinite(alpha)
    run.optimizer.param_groups[0]['lr'] = .0125
    seen = []
    run.hook = lambda: seen.append((sampler.step_size.clone(), sampler._alpha.clone()))
    step(run)
    assert len(seen) == 1  # the proposal evaluation sees the transition's step and history
    close(seen[0][0], adapted*.25)
    assert torch.equal(seen[0][1], alpha)
    close(run.optimizer.last_result.stats['step_size'], adapted*.25)


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('adapt_steps', [0, None])
@pytest.mark.parametrize('group', [0, 1])
def test_lip_group_lr_change_scales_only_that_groups_effective_step(name, adapt_steps, group):
    run = make(name, lrs=(.05, .1), adapt_steps=adapt_steps)
    for _ in range(3):
        step(run)
    sampler = run.optimizer._sampler
    planned = sampler.step_size*sampler.mobility  # per-coordinate step of the next transition
    run.optimizer.param_groups[group]['lr'] *= .25
    step(run)
    used = run.optimizer.last_result.stats['step_size']*sampler.mobility
    ratio = torch.ones(7, dtype=DT)
    ratio[:4] = .25 if group == 0 else 1.
    ratio[4:] = .25 if group == 1 else 1.
    close(used, planned*ratio)


@pytest.mark.parametrize('name', LIP)
def test_lip_lr_increase_is_clamped_at_max_step_size(name):
    run = make(name, lr=.05, adapt_steps=0, max_step_size=.2)
    step(run)
    run.optimizer.param_groups[0]['lr'] = 1.
    step(run)
    assert run.optimizer.last_result.stats['step_size'] == .2
    # Documented consequence: a later decrease scales the clamped step, it does not undo the increase.
    run.optimizer.param_groups[0]['lr'] = .05
    step(run)
    close(run.optimizer.last_result.stats['step_size'], .01)


@pytest.mark.parametrize('name', LIP)
def test_lip_clamped_group_zero_increase_lowers_the_other_groups_step(name):
    # lr ratios between groups are always honoured, so at the cap group 0 cannot rise and group 1 falls.
    run = make(name, lrs=(.05, .1), adapt_steps=0, max_step_size=.2)
    step(run)
    sampler = run.optimizer._sampler
    close(sampler.step_size*sampler.mobility, torch.tensor([.05]*4+[.1]*3, dtype=DT))
    run.optimizer.param_groups[0]['lr'] = 1.  # asks for 20 times the step; the cap allows 4 times
    step(run)
    assert run.optimizer.last_result.stats['step_size'] == .2
    close(run.optimizer.last_result.stats['step_size']*sampler.mobility, torch.tensor([.2]*4+[.02]*3, dtype=DT))
    # Below the cap the same edit moves group 0 alone.
    free = make(name, lrs=(.05, .1), adapt_steps=0)
    step(free)
    free.optimizer.param_groups[0]['lr'] = .1
    step(free)
    close(free.optimizer.last_result.stats['step_size']*free.optimizer._sampler.mobility, torch.full((7,), .1, dtype=DT))


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('stored', [True, False])
def test_checkpoint_with_legacy_bool_adapt_steps_continues_as_the_equal_int(name, stored):
    # Before bool adapt_steps was rejected an adapter could be built, and saved, with one.
    run = make(name, lrs=(.05, .1), adapt_steps=int(stored))
    for _ in range(3):
        step(run)
    saved = run.optimizer.state_dict()
    expected = trajectory(run, 4)
    saved['lafa']['sampler_kwargs']['adapt_steps'] = stored
    if 'adapt_steps' in saved['lafa']['sampler']['state']:
        saved['lafa']['sampler']['state']['adapt_steps'] = stored
    other = resumed(name, saved)
    assert_nested(trajectory(other, 4), expected)
    assert type(other.optimizer.sampler_kwargs['adapt_steps']) is int
    assert saved['lafa']['sampler_kwargs']['adapt_steps'] is stored  # the caller's checkpoint is not edited


@pytest.mark.parametrize('name', LIP)
def test_lip_step_follows_lr_scheduler_ratio(name):
    run = make(name, lr=.08, adapt_steps=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(run.optimizer, gamma=.5)
    for epoch in range(5):
        step(run)
        assert scheduler.get_last_lr() == [.08*.5**epoch]
        close(run.optimizer.last_result.stats['step_size'], .08*.5**epoch)
        scheduler.step()


def test_lip_lr_change_that_underflows_the_step_is_rejected_and_rolled_back():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    optimizer = lafa.LipULA([p], lr=.05, adapt_steps=0, generator=torch.Generator().manual_seed(5))

    def closure():
        optimizer.zero_grad(); value = p.square().sum()/2; value.backward(); return value
    optimizer.step(closure)
    before, sampler = p.detach().clone(), optimizer._sampler.state_dict()
    optimizer.param_groups[0]['lr'] = 1e-46  # .05*(1e-46/.05) is zero in float32
    with pytest.raises(ValueError, match='underflows'):
        optimizer.step(closure)
    assert torch.equal(p, before)
    assert_nested(optimizer._sampler.state_dict(), sampler)
    assert optimizer.state_dict()['lafa']['sampler_lr'] == .05


EDITS = {'none': (1., False), 'applied': (.25, True), 'pending': (.25, False),
         'clamped': (20., True), 'pending_clamped': (20., False)}


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('adapt_steps', [0, None])
@pytest.mark.parametrize('edit', list(EDITS))
def test_lip_checkpoint_resume_equals_uninterrupted_run(name, adapt_steps, edit):
    ratio, step_before_saving = EDITS[edit]
    run = make(name, lrs=(.05, .1), adapt_steps=adapt_steps, max_step_size=.2)
    for _ in range(3):
        step(run)
    planned = run.optimizer._sampler.step_size.clone()
    run.optimizer.param_groups[0]['lr'] = .05*ratio
    used = trajectory(run, 1)[0][3] if step_before_saving else None  # the change reaches the sampler before saving
    saved = run.optimizer.state_dict()
    expected = trajectory(run, 4)
    close(expected[0][3] if used is None else used, (planned*ratio).clamp(max=.2))
    # The saved reference is the lr the saved sampler step corresponds to, not necessarily the saved lr.
    assert saved['lafa']['sampler_lr'] == (.05*ratio if step_before_saving else .05)
    # After a clamped increase lr (1.) exceeds max_step_size (.2); the resumed sampler must still build.
    assert_nested(trajectory(resumed(name, saved), 4), expected)


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('legacy', [False, True])
def test_lip_lr_edit_between_load_and_first_resumed_step_applies_the_ratio(name, legacy):
    run = make(name, lrs=(.05, .1))
    for _ in range(3):
        step(run)
    saved = run.optimizer.state_dict()
    if legacy:
        del saved['lafa']['sampler_lr']  # written before lr tracking: the saved lr is the reference
    planned = run.optimizer._sampler.step_size.clone()
    run.optimizer.param_groups[0]['lr'] = .0125
    expected = trajectory(run, 4)
    other = resumed(name, saved)
    other.optimizer.param_groups[0]['lr'] = .0125
    actual = trajectory(other, 4)
    close(actual[0][3], planned*.25)
    assert_nested(actual, expected)


@pytest.mark.parametrize('name', ADAPTERS)
def test_checkpoint_without_tracked_lr_loads_as_unchanged(name):
    run = make(name, **CASES[name])
    for _ in range(3):
        step(run)
    saved = run.optimizer.state_dict()
    # Schema-1 checkpoints written before lr tracking have no such key.
    assert saved['lafa'].pop('sampler_lr') == (None if name == 'ULA' else run.optimizer.param_groups[0]['lr'])
    expected = trajectory(run, 4)
    assert_nested(trajectory(resumed(name, saved), 4), expected)


@pytest.mark.parametrize('reference', [0., -.05, float('nan'), float('inf'), '0.05', True, 1])
def test_checkpoint_with_invalid_tracked_lr_is_rejected_before_any_change(reference):
    run = make('LipULA', lrs=(.05, .1))
    for _ in range(2):
        step(run)
    saved = run.optimizer.state_dict()
    step(run)
    before, sampler = run.optimizer.state_dict(), run.optimizer._sampler
    saved['lafa']['sampler_lr'] = reference
    with pytest.raises(ValueError, match='sampler_lr'):
        run.optimizer.load_state_dict(saved)
    assert_nested(run.optimizer.state_dict(), before)
    assert run.optimizer._sampler is sampler


@pytest.mark.parametrize('name', LIP)
def test_lip_failed_step_restores_tracked_lr_and_step(name):
    run, twin = make(name, lrs=(.05, .1)), make(name, lrs=(.05, .1))
    for _ in range(3):
        assert torch.equal(step(run), step(twin))
    for each in (run, twin):
        each.optimizer.param_groups[0]['lr'] = .0125
    before, start = run.optimizer.state_dict(), run.flat()
    planned = run.optimizer._sampler.step_size.clone()
    assert before['lafa']['sampler_lr'] == .05

    def fail():
        if not torch.equal(run.flat(), start):
            raise RuntimeError('proposal failed')
    run.hook = fail
    with pytest.raises(RuntimeError, match='proposal failed'):
        step(run)
    run.hook = None
    assert_nested(run.optimizer.state_dict(), before)
    assert torch.equal(run.optimizer._sampler.step_size, planned)
    # The retry applies the ratio exactly once, as the twin that never failed.
    for index in range(3):
        assert torch.equal(step(run), step(twin))
        assert_same(run, twin)
        if index == 0:
            close(run.optimizer.last_result.stats['step_size'], planned*.25)
    assert_nested(run.optimizer.state_dict(), twin.optimizer.state_dict())


# Cached potential/gradient reuse.

@pytest.mark.parametrize('name', CLOSURE_ADAPTERS)
def test_same_closure_object_costs_one_call_per_step_with_bit_identical_trajectory(name):
    reuse, fresh = make(name, **CASES[name]), make(name, **CASES[name])
    accepted = set()
    for _ in range(10):
        assert torch.equal(step(reuse), step(fresh, fresh=True))
        assert_same(reuse, fresh)
        if reuse.optimizer.last_result.accepted is not None:
            accepted.add(bool(reuse.optimizer.last_result.accepted))
    assert reuse.calls == 11 and fresh.calls == 20
    assert accepted == ({True, False} if 'MALA' in name else set())
    assert_nested(reuse.optimizer.state_dict(), fresh.optimizer.state_dict())


@pytest.mark.parametrize('name', CLOSURE_ADAPTERS)
def test_external_parameter_edit_forces_reevaluation(name):
    reuse, fresh = make(name, **CASES[name]), make(name, **CASES[name])
    for _ in range(3):
        step(reuse); step(fresh, fresh=True)
    for each in (reuse, fresh):
        with torch.no_grad():
            each.b[1] += .5
    calls = reuse.calls
    assert torch.equal(step(reuse), step(fresh, fresh=True))
    assert reuse.calls == calls+2
    assert_same(reuse, fresh)
    assert torch.equal(step(reuse), step(fresh, fresh=True))
    assert reuse.calls == calls+3
    assert_same(reuse, fresh)


def test_sign_of_zero_counts_as_a_parameter_edit():
    # torch.equal alone would take -0. for the cached +0. state and write +0. back.
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    optimizer = lafa.MALA([p], lr=1e30, temperature=1e30, generator=torch.Generator().manual_seed(3))
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(); value = p.square().sum()/2; value.backward(); return value
    optimizer.step(closure)  # nonfinite proposal: rejected, parameters stay +0.
    assert not optimizer.last_result.accepted and not torch.signbit(p).any()
    with torch.no_grad():
        p.neg_()
    calls = 0
    optimizer.step(closure)
    assert calls == 2 and not optimizer.last_result.accepted
    assert torch.signbit(p).all() and torch.signbit(optimizer.last_result.gradient).all()


@pytest.mark.parametrize('name', CLOSURE_ADAPTERS)
def test_invalidate_cache_forces_reevaluation_after_data_change(name):
    reuse, fresh, stale = (make(name, **CASES[name]) for _ in range(3))
    for _ in range(3):
        step(reuse); step(fresh, fresh=True); step(stale)
    for each in (reuse, fresh, stale):
        each.shift = 1.  # data behind the same closure object
    reuse.optimizer.invalidate_cache()
    calls = reuse.calls
    assert torch.equal(step(reuse), step(fresh, fresh=True))
    assert reuse.calls == calls+2
    assert_same(reuse, fresh)
    assert torch.equal(step(reuse), step(fresh, fresh=True))
    assert reuse.calls == calls+3
    assert_same(reuse, fresh)
    # Contract check: without the call the step starts from the gradient of the old data.
    step(stale)
    assert stale.calls == calls+1
    assert not torch.equal(stale.grads(), fresh.grads()) or not torch.equal(stale.flat(), fresh.flat())


@pytest.mark.parametrize('name', CLOSURE_ADAPTERS)
def test_failed_step_restores_cache_bookkeeping(name):
    run, twin = make(name, **CASES[name]), make(name, **CASES[name])
    for _ in range(2):
        step(run); step(twin)

    def broken():
        run.closure()
        raise RuntimeError('broken closure')
    with pytest.raises(RuntimeError, match='broken closure'):
        run.optimizer.step(broken)
    calls = run.calls
    assert torch.equal(step(run), step(twin))
    assert run.calls == calls+1  # the last successful step's cache is still attributed to run.closure
    assert_same(run, twin)
    # An invalidation requested before a failed step survives its rollback.
    run.optimizer.invalidate_cache()
    start = run.flat()

    def fail():
        if not torch.equal(run.flat(), start):
            raise RuntimeError('proposal failed')
    run.hook = fail
    with pytest.raises(RuntimeError, match='proposal failed'):
        step(run)
    run.hook = None
    calls = run.calls
    assert torch.equal(step(run), step(twin))
    assert run.calls == calls+2
    assert_same(run, twin)


@pytest.mark.parametrize('name', CLOSURE_ADAPTERS)
def test_load_state_dict_resets_cache(name):
    run, twin = make(name, **CASES[name]), make(name, **CASES[name])
    for _ in range(2):
        step(run); step(twin)
    assert run.optimizer._cached_closure is run.closure
    run.optimizer.load_state_dict(run.optimizer.state_dict())
    # The rebuilt sampler evaluates anyway; the reset itself is only visible on the bookkeeping.
    assert run.optimizer._cached_closure is None
    calls = run.calls
    assert torch.equal(step(run), step(twin))
    assert run.calls == calls+2
    assert_same(run, twin)


def test_ula_adapter_keeps_one_call_per_step_and_ignores_invalidate_cache():
    run, twin = make('ULA', **CASES['ULA']), make('ULA', **CASES['ULA'])
    for _ in range(10):
        run.optimizer.invalidate_cache()
        assert torch.equal(step(run), step(twin, fresh=True))
        assert_same(run, twin)
    assert run.calls == twin.calls == 10
