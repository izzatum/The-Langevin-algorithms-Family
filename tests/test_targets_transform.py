"""positive_transform temperature scaling and zero-size ModulePotential parameters.

Independent oracles are closed forms in the unconstrained coordinate x=log(z).
"""
import math
from fractions import Fraction

import pytest
import torch

from lafa.samplers import MALA
from lafa.targets import ModulePotential, positive_transform, value_and_grad

DT = torch.float64
SHAPES = [(3,), (5, 3)]
RATE = torch.tensor([.5, 1., 2.], dtype=DT)
CONCENTRATION = torch.tensor([1.5, 2., 3.5], dtype=DT)


def gamma(z):
    """Independent Gamma(concentration, rate) potential on z>0."""
    return (RATE.to(z)*z - (CONCENTRATION.to(z)-1)*z.log()).sum(-1)


def position(shape, dtype=DT, seed=5):
    return torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', SHAPES)
def test_default_transform_is_bit_identical_to_the_original_formula(dtype, shape):
    x = position(shape, dtype)
    # The complete definition before the temperature keyword existed.
    original = lambda x: gamma(x.exp()) - x.sum(-1)
    transformed = positive_transform(gamma)
    expected_value, expected_gradient = value_and_grad(original, x)
    value, gradient = value_and_grad(transformed, x)
    assert torch.equal(transformed(x), original(x))
    assert torch.equal(value, expected_value) and torch.equal(gradient, expected_gradient)
    assert value.dtype == dtype and gradient.dtype == dtype
    # Closed form in x, independent of both lambdas.
    torch.testing.assert_close(value, (RATE.to(x)*x.exp() - CONCENTRATION.to(x)*x).sum(-1))
    torch.testing.assert_close(gradient, RATE.to(x)*x.exp() - CONCENTRATION.to(x))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('temperature', [1., 1])
def test_explicit_unit_temperature_is_bit_identical_to_the_default(dtype, shape, temperature):
    x = position(shape, dtype)
    default, explicit = positive_transform(gamma), positive_transform(gamma, temperature=temperature)
    assert torch.equal(explicit(x), default(x))
    assert all(torch.equal(a, b) for a, b in zip(value_and_grad(explicit, x), value_and_grad(default, x)))


@pytest.mark.parametrize('temperature', [.5, 2., 3.7, 2, Fraction(7, 2)])
@pytest.mark.parametrize('shape', SHAPES)
def test_tempered_density_keeps_the_jacobian_untempered(shape, temperature):
    x = position(shape)
    scale = float(temperature)
    transformed = positive_transform(gamma, temperature=temperature)
    # -transformed/T is the sampled log density: log pi(exp(x))/T plus log|dz/dx|.
    torch.testing.assert_close(transformed(x)/scale, gamma(x.exp())/scale - x.sum(-1))
    value, gradient = value_and_grad(transformed, x)
    assert value.shape == x.shape[:-1] and gradient.shape == x.shape
    torch.testing.assert_close(value/scale, (RATE*x.exp() - (CONCENTRATION-1)*x).sum(-1)/scale - x.sum(-1))
    torch.testing.assert_close(gradient/scale, (RATE*x.exp() - (CONCENTRATION-1))/scale - 1)


def test_temperature_changes_only_the_jacobian_term():
    x = position((5, 3))
    difference = positive_transform(gamma, temperature=3.)(x) - positive_transform(gamma)(x)
    torch.testing.assert_close(difference, -2*x.sum(-1))


def test_temperature_preserves_float32_positions():
    x = position((5, 3), torch.float32)
    transformed = positive_transform(gamma, temperature=2.)
    value, gradient = value_and_grad(transformed, x)
    reference_value, reference_gradient = value_and_grad(transformed, x.double())
    assert value.dtype == torch.float32 and gradient.dtype == torch.float32
    torch.testing.assert_close(value.double(), reference_value, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gradient.double(), reference_gradient, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('temperature', [0., -0., 0, -1., float('inf'), -float('inf'), float('nan'), True, False,
                                         None, '2', 2j, [2.], torch.tensor(2.),
                                         pytest.param(10**400, id='overflowing-int'),
                                         pytest.param(Fraction(1, 10**400), id='underflowing-fraction')])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match='temperature must be a positive finite real number'):
        positive_transform(gamma, temperature=temperature)


def test_temperature_is_keyword_only():
    with pytest.raises(TypeError):
        positive_transform(gamma, 2.)


def test_mala_tempered_positive_target_recovers_the_tempered_mean():
    # exp(-z/T) on z>0 is exponential with mean T and variance T^2. The chain
    # endpoints are independent, so the standard error of their mean is
    # T/sqrt(chains)=1/32 and the tolerance is eight standard errors. Tempering
    # the Jacobian as well would sample a chi-square(1) law with mean 1, which
    # is 32 standard errors from T=2.
    temperature, chains = 2., 4096
    target = positive_transform(lambda z: z.sum(-1), temperature=temperature)
    sampler = MALA(target, torch.zeros(chains, 1, dtype=DT), step_size=.25, temperature=temperature,
                   generator=torch.Generator().manual_seed(11))
    sampler.advance(120)
    z = sampler.position.exp()
    assert abs(z.mean().item() - temperature) < 8*temperature/math.sqrt(chains)


class ZeroSized(torch.nn.Module):
    """Two empty parameters around a regular one; forward uses all three."""
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Parameter(torch.empty(0, dtype=DT))
        self.w = torch.nn.Parameter(torch.tensor([.3, -.7, 1.1], dtype=DT))
        self.tail = torch.nn.Parameter(torch.empty(0, 2, dtype=DT))

    def forward(self):
        return self.w.sin() + self.head.sum() + self.tail.sum()


@pytest.mark.parametrize('mode', ['vmap', 'loop'])
@pytest.mark.parametrize('chunk', [None, 2])
def test_module_with_zero_size_parameters_constructs_and_evaluates(mode, chunk):
    model = ZeroSized()
    target = ModulePotential(model, lambda y: y.square().sum()/2, batch_mode=mode, chunk_size=chunk)
    assert target.numel == 3
    assert [(s.name, tuple(s.shape), s.start, s.stop) for s in target.parameter_spec] == [
        ('head', (0,), 0, 0), ('w', (3,), 0, 3), ('tail', (0, 2), 3, 3)]
    assert torch.equal(target.flatten_parameters(), model.w.detach())
    for shape in SHAPES:
        x = position(shape, seed=3)
        expected_value, expected_gradient = x.sin().square().sum(-1)/2, x.sin()*x.cos()
        torch.testing.assert_close(target(x), expected_value)
        for value, gradient in (target.value_and_grad(x), value_and_grad(target, x)):
            assert value.shape == x.shape[:-1] and gradient.shape == x.shape
            torch.testing.assert_close(value, expected_value)
            torch.testing.assert_close(gradient, expected_gradient)
    assert all(p.grad is None for p in model.parameters())


def test_zero_size_view_of_parameter_storage_is_not_an_alias():
    model = ZeroSized()
    model.view = torch.nn.Parameter(model.w.detach()[:0])
    assert model.view.untyped_storage().data_ptr() == model.w.untyped_storage().data_ptr()
    target = ModulePotential(model, lambda y: y.square().sum()/2)
    x = position((3,), seed=3)
    assert target.numel == 3 and len(target.parameter_spec) == 4
    torch.testing.assert_close(target(x), x.sin().square().sum()/2)


def test_nonempty_aliases_are_still_rejected_beside_zero_size_parameters():
    model = ZeroSized()
    model.alias = torch.nn.Parameter(model.w.detach())
    with pytest.raises(ValueError, match='aliased parameter storage'):
        ModulePotential(model, lambda y: y.square().sum())
    del model.head, model.tail
    with pytest.raises(ValueError, match='aliased parameter storage'):
        ModulePotential(model, lambda y: y.square().sum())


def test_module_with_only_zero_size_parameters_has_no_coordinates():
    model = ZeroSized()
    del model.w
    with pytest.raises(ValueError, match='module must have parameters'):
        ModulePotential(model, lambda y: y.sum())
