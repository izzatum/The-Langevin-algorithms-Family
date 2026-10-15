"""User callbacks may consume their arguments without changing retained state."""
import math

import pytest
import torch

from lafa.specialized import (
    AnchoredLangevin, BirthDeathLangevin, HOLA, MYIPLA,
    NoiseAnnealedLangevin, PIPGLA, PIPULA, SALEnKF,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def gaussian(x):
    return x.square().sum(-1) / 2


def test_myipla_inplace_exact_laplace_prox_preserves_old_state():
    x, theta = tensor([[2.], [-3.]]), tensor([.4])

    def smooth(t, x):
        return x.sum(-1) * 0 + t.sum() * 0

    def prox(t, x, smoothing):
        x.copy_(x.sign() * (x.abs() - smoothing).clamp_min(0))
        return t.expand(len(x), -1), x

    sampler = MYIPLA(smooth, prox, theta, x, .1, smoothing=.5)
    result = sampler.step(theta_noise=torch.zeros_like(theta), particle_noise=torch.zeros_like(x))
    torch.testing.assert_close(result.particles, tensor([[1.9], [-2.9]]))
    torch.testing.assert_close(result.theta, theta)
    torch.testing.assert_close(x, tensor([[2.], [-3.]]))


@pytest.mark.parametrize('method', [MYIPLA, PIPULA, PIPGLA])
def test_prox_mutation_then_error_preserves_sampler_state(method):
    theta, x = tensor([.4]), tensor([[2.], [-3.]])

    def prox(t, x, smoothing):
        t.zero_()
        x.zero_()
        raise RuntimeError('prox failure')

    smooth = lambda t, x: gaussian(x) + t.square().sum() / 2
    sampler = method(prox, theta, x) if method is PIPULA else method(smooth, prox, theta, x)
    with pytest.raises(RuntimeError, match='prox failure'):
        sampler.step(theta_noise=torch.zeros_like(theta), particle_noise=torch.zeros_like(x))
    torch.testing.assert_close(sampler.theta, theta)
    torch.testing.assert_close(sampler.position, x)
    assert sampler.iteration == 0


@pytest.mark.parametrize('method', [MYIPLA, PIPGLA])
def test_inplace_smooth_callback_preserves_parameter_gradients(method):
    theta, x = tensor([.4]), tensor([[2.], [-3.]])

    def smooth(t, x):
        return x.square_().sum(-1) / 2 + t.square_().sum() / 2

    def identity(t, x, smoothing):
        return t.expand(len(x), -1), x

    sampler = method(smooth, identity, theta, x, .1)
    result = sampler.step(theta_noise=torch.zeros_like(theta), particle_noise=torch.zeros_like(x))
    torch.testing.assert_close(result.theta, .9 * theta)
    torch.testing.assert_close(result.particles, .9 * x)
    assert result.theta.grad_fn is None and result.particles.grad_fn is None


@pytest.mark.parametrize('failure', [False, True])
def test_noise_annealing_inplace_gaussian_score(failure):
    x = tensor([2.])

    def score(x, sigma):
        x.neg_()
        if failure:
            raise RuntimeError('score failure')
        return x

    sampler = NoiseAnnealedLangevin(score, x, [1.], step_size=.1)
    if failure:
        with pytest.raises(RuntimeError, match='score failure'):
            sampler.step(noise=torch.zeros_like(x))
        torch.testing.assert_close(sampler.position, x)
        assert sampler.iteration == 0
    else:
        torch.testing.assert_close(sampler.step(noise=torch.zeros_like(x)).position, .95 * x)


@pytest.mark.parametrize('failure', [False, True])
def test_anchored_inplace_potential_preserves_constructor_input_and_state(failure):
    x = tensor([2.])

    def potential(z):
        return z.square_().sum(-1) / 2

    sampler = AnchoredLangevin(potential, gaussian, x, .1)
    torch.testing.assert_close(x, tensor([2.]))
    if failure:
        def fail(z):
            z.zero_()
            raise RuntimeError('potential failure')
        sampler.potential = fail
        with pytest.raises(RuntimeError, match='potential failure'):
            sampler.step(noise=torch.zeros_like(x))
        torch.testing.assert_close(sampler.position, x)
        torch.testing.assert_close(sampler.operational_time, tensor(0.))
        assert sampler.iteration == 0
    else:
        torch.testing.assert_close(sampler.step(noise=torch.zeros_like(x)).position, .9 * x)


@pytest.mark.parametrize('failure', [False, True])
def test_birth_death_rates_inplace_potential_preserves_public_and_owned_particles(failure):
    x = tensor([[1.], [3.]])

    def potential(z):
        z.square_()
        if failure:
            raise RuntimeError('potential failure')
        return z.sum(-1) / 2

    sampler = BirthDeathLangevin(potential, x)
    for particles in (None, x):
        if failure:
            with pytest.raises(RuntimeError, match='potential failure'):
                sampler.rates(particles)
        else:
            expected = BirthDeathLangevin(gaussian, x).rates()
            torch.testing.assert_close(sampler.rates(particles), expected)
        torch.testing.assert_close(x, tensor([[1.], [3.]]))
        torch.testing.assert_close(sampler.position, x)


class InplaceQuadraticDerivatives:
    """Exact U=||x||² derivatives, consuming every supplied buffer."""
    def __init__(self, failure=None):
        self.failure = failure

    def _fail(self, name, x):
        if self.failure == name:
            x.zero_()
            raise RuntimeError('derivative failure')

    def gradient(self, x):
        self._fail('gradient', x)
        return x.mul_(2)

    def grad_laplacian(self, x):
        self._fail('grad_laplacian', x)
        return x.zero_()

    def spectral_norm(self, x):
        self._fail('spectral_norm', x)
        return x.fill_(2).max()

    def hvp(self, x, vector):
        self._fail('hvp', x)
        x.zero_()
        return vector.mul_(2)


def test_hola_inplace_derivative_provider_preserves_all_vectors():
    x, z, za = tensor([.7, -1.]), tensor([.2, -.4]), tensor([-.1, .3])
    h, a = .1, 2.
    sampler = HOLA(lambda y: y.square().sum(), x, h, tamed=False,
                   derivatives=InplaceQuadraticDerivatives())
    result = sampler.step(noise=z, auxiliary_noise=za)
    expected = (1-h*a+h*h*a*a/2)*x + math.sqrt(2*h)*((1-h*a/2)*z+h*a/math.sqrt(12)*za)
    torch.testing.assert_close(result.position, expected)
    torch.testing.assert_close(z, tensor([.2, -.4]))
    torch.testing.assert_close(za, tensor([-.1, .3]))


@pytest.mark.parametrize('callback', ['gradient', 'grad_laplacian', 'spectral_norm', 'hvp'])
def test_hola_provider_mutation_then_error_preserves_sampler_state(callback):
    x = tensor([.7, -1.])
    sampler = HOLA(gaussian, x, derivatives=InplaceQuadraticDerivatives(callback))
    with pytest.raises(RuntimeError, match='derivative failure'):
        sampler.step(noise=torch.zeros_like(x), auxiliary_noise=torch.zeros_like(x))
    torch.testing.assert_close(sampler.position, x)
    assert sampler.iteration == 0


@pytest.mark.parametrize('failure', [False, True])
def test_hola_dense_inplace_potential_preserves_higher_derivatives_and_state(failure):
    x = tensor([.7, -1.])

    def potential(z):
        z.square_()
        if failure:
            raise RuntimeError('dense potential failure')
        return z.sum() / 2

    sampler = HOLA(potential, x, .1, tamed=False)
    if failure:
        with pytest.raises(RuntimeError, match='dense potential failure'):
            sampler.step(noise=torch.zeros_like(x), auxiliary_noise=torch.zeros_like(x))
        torch.testing.assert_close(sampler.position, x)
        assert sampler.iteration == 0
    else:
        gradient, hvp, norm, laplacian = sampler._derivatives(x)
        torch.testing.assert_close(gradient, x)
        torch.testing.assert_close(hvp(x), x)
        torch.testing.assert_close(laplacian, torch.zeros_like(x))
        torch.testing.assert_close(norm, tensor(1.))
        result = sampler.step(noise=torch.zeros_like(x), auxiliary_noise=torch.zeros_like(x))
        torch.testing.assert_close(result.position, (1-.1+.1**2/2)*x)
        torch.testing.assert_close(x, tensor([.7, -1.]))


def filtering_model(inplace):
    if inplace:
        transition = lambda x, t: x.mul_(t.add_(.25)[0])
        covariance = lambda t: t.exp_().reshape(1, 1)
        observation = lambda t: t.add_(1).reshape(1, 1)
        variance = lambda t: t.mul_(2).exp_().reshape(1, 1)
    else:
        transition = lambda x, t: x * (t[0] + .25)
        covariance = lambda t: t.exp().reshape(1, 1)
        observation = lambda t: (t + 1).reshape(1, 1)
        variance = lambda t: (2*t).exp().reshape(1, 1)
    return SALEnKF(transition, covariance, observation, variance, gaussian,
                   tensor([[.2], [-.4], [.7]]), tensor([.3]), inner_steps=3,
                   burnin=1, gamma=.001, generator=torch.Generator().manual_seed(52))


def test_filtering_inplace_callbacks_preserve_predictive_score_autograd():
    sampler = filtering_model(True)
    ancestors, y = tensor([[-1.], [.2], [2.]]).requires_grad_(), tensor([.8])
    original = ancestors.detach().clone()
    score = sampler.predictive_score(y, ancestors)
    t = .3
    f, h, c, v = t+.25, 1+t, math.exp(t), math.exp(2*t)
    residual = y - h*f*original[:, 0]
    variance = h*h*c+v
    derivative = (2*h+h*h)*c+2*v
    expected = ((residual*(h+f)*original[:, 0])/variance
                + .5*derivative*(residual.square()/variance**2-1/variance)).mean()
    torch.testing.assert_close(score, expected.reshape(1))
    torch.testing.assert_close(ancestors, original)
    torch.testing.assert_close(sampler.theta, tensor([.3]))
    assert ancestors.grad is None and score.grad_fn is None


def test_filtering_inplace_transition_preserves_backward_history_pool():
    sampler = filtering_model(True)
    pool, current = tensor([[-1.], [1.]]), tensor([[.5], [-.2]])
    original = pool.clone()
    actual = sampler.backward_probabilities(current, pool, block_size=1)
    expected = (-.5*(current-.55*original.T).square()/math.exp(.3)).softmax(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pool, original)
    torch.testing.assert_close(sampler.theta, tensor([.3]))


def test_filtering_inplace_callbacks_match_two_stage_pure_updates():
    pure, inplace = filtering_model(False), filtering_model(True)
    for observation in (tensor([.4]), tensor([.9])):
        expected, actual = pure.step(observation), inplace.step(observation)
        torch.testing.assert_close(actual.particles, expected.particles)
        torch.testing.assert_close(actual.theta, expected.theta)
        torch.testing.assert_close(inplace.pool, pure.pool)
        assert actual.particles.grad_fn is None and inplace.pool.grad_fn is None


@pytest.mark.parametrize('callback', ['transition', 'process_covariance', 'observation_matrix', 'observation_covariance'])
def test_filtering_callback_mutation_then_error_preserves_stage_state(callback):
    sampler = filtering_model(False)
    sampler.step(tensor([.4]))
    theta, position, pool = sampler.theta, sampler.position, sampler.pool

    def fail(*arguments):
        with torch.no_grad():
            for argument in arguments:
                argument.zero_()
        raise RuntimeError('filter callback failure')

    setattr(sampler, callback, fail)
    with pytest.raises(RuntimeError, match='filter callback failure'):
        sampler.step(tensor([.9]))
    torch.testing.assert_close(sampler.theta, theta)
    torch.testing.assert_close(sampler.position, position)
    torch.testing.assert_close(sampler.pool, pool)
    assert sampler.iteration == 1


def test_filtering_public_backward_error_preserves_pool_and_theta():
    sampler = filtering_model(False)
    pool = tensor([[-1.], [1.]])

    def fail(x, t):
        x.zero_()
        t.zero_()
        raise RuntimeError('backward failure')

    sampler.transition = fail
    with pytest.raises(RuntimeError, match='backward failure'):
        sampler.backward_probabilities(tensor([[.5]]), pool)
    torch.testing.assert_close(pool, tensor([[-1.], [1.]]))
    torch.testing.assert_close(sampler.theta, tensor([.3]))
