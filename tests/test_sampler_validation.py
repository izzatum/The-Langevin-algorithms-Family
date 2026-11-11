"""Lip samplers validate adapt_steps; 'adapting' reports whether the next transition adapts."""
import pytest
import torch
from lafa import optim, samplers

LIP = ['LipULA', 'LipMALA']
START = torch.linspace(-1.5, 1.5, 12, dtype=torch.float64).reshape(6, 2)


def target(x):
    # Non-quadratic: the local Lipschitz estimate, hence h, changes with every move.
    return (x.square()/2 + torch.log(torch.cosh(x))).sum(-1)


def make(name, **kwargs):
    return getattr(samplers, name)(target, START, step_size=.05,
                                  generator=torch.Generator().manual_seed(11), **kwargs)


def run(sampler, steps):
    """Per step: 'adapting', the step size used, and the one left for the next transition."""
    rows = []
    for _ in range(steps):
        used = sampler.step_size.clone()
        stats = sampler.step().stats
        assert torch.equal(stats['step_size'], used)
        assert torch.equal(stats['next_step_size'], sampler.step_size)
        rows.append((bool(stats['adapting']), used, sampler.step_size.clone()))
    return rows


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('value', [True, False])
def test_adapt_steps_rejects_bool(name, value):
    with pytest.raises(ValueError, match='adapt_steps must be a nonnegative integer or None'):
        make(name, adapt_steps=value)


@pytest.mark.parametrize('name', LIP)
def test_optimizer_adapter_rejects_bool_adapt_steps(name):
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    def closure():
        parameter.grad = None
        value = parameter.square().sum()/2
        value.backward()
        return value
    # The adapter builds its sampler on the first step; either place may reject.
    with pytest.raises(ValueError, match='adapt_steps must be a nonnegative integer or None'):
        optimizer = getattr(optim, name)([parameter], lr=.05, adapt_steps=True,
                                         generator=torch.Generator().manual_seed(11))
        optimizer.step(closure)


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('value', [-1, 2., '2'])
def test_adapt_steps_rejects_negative_and_non_integer_values(name, value):
    with pytest.raises(ValueError, match='adapt_steps must be a nonnegative integer or None'):
        make(name, adapt_steps=value)


@pytest.mark.parametrize('name', LIP)
@pytest.mark.parametrize('value', [None, 0, 3])
def test_adapt_steps_accepts_none_and_nonnegative_integers(name, value):
    sampler = make(name, adapt_steps=value)
    assert sampler.adapt_steps == value and type(sampler.adapt_steps) is type(value)


@pytest.mark.parametrize('name', LIP)
def test_adapt_steps_two_reports_true_false_false_and_freezes_after_step_two(name):
    limited, free = make(name, adapt_steps=2), make(name)
    rows, free_rows = run(limited, 5), run(free, 5)
    assert [adapting for adapting, _, _ in rows[:3]] == [True, False, False]
    assert not any(adapting for adapting, _, _ in rows[3:])
    # Transitions 1 and 2 both adapt, exactly as the never-frozen twin does.
    for (_, used, left), (_, _, free_left) in zip(rows[:2], free_rows[:2]):
        assert not torch.equal(left, used) and torch.equal(left, free_left)
    # 'adapting' False at step 2 means its next_step_size is final.
    frozen = rows[1][2]
    for _, used, left in rows[2:]:
        assert torch.equal(used, frozen) and torch.equal(left, frozen)
    # Step 3 starts from the same state in both samplers, and only the twin changes h.
    assert not torch.equal(free_rows[2][2], frozen)


@pytest.mark.parametrize('name', LIP)
def test_adapt_steps_one_adapts_once_but_never_reports_adapting(name):
    limited, free = make(name, adapt_steps=1), make(name)
    rows, free_rows = run(limited, 4), run(free, 4)
    assert [adapting for adapting, _, _ in rows] == [False]*4
    # The flag describes the next transition, not the one that produced it.
    assert not torch.equal(rows[0][2], rows[0][1]) and torch.equal(rows[0][2], free_rows[0][2])
    for _, used, left in rows[1:]:
        assert torch.equal(used, rows[0][2]) and torch.equal(left, rows[0][2])
    assert not torch.equal(free_rows[1][2], rows[0][2])


@pytest.mark.parametrize('name', LIP)
def test_adapt_steps_none_reports_adapting_on_every_step(name):
    rows = run(make(name), 6)
    assert [adapting for adapting, _, _ in rows] == [True]*6
    assert all(not torch.equal(left, used) for _, used, left in rows)


@pytest.mark.parametrize('name,fixed', [('LipULA', 'ULA'), ('LipMALA', 'MALA')])
def test_adapt_steps_zero_never_adapts_and_is_the_fixed_step_kernel(name, fixed):
    limited, plain = make(name, adapt_steps=0), make(fixed)
    initial = limited.step_size.clone()
    rows = run(limited, 5)
    assert [adapting for adapting, _, _ in rows] == [False]*5
    assert all(torch.equal(used, initial) and torch.equal(left, initial) for _, used, left in rows)
    plain.advance(5)
    assert torch.equal(limited.position, plain.position)
    assert not torch.equal(limited.position, START)
