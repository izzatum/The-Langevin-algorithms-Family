"""Independent reference and calibration contracts, not convergence guarantees."""
from pathlib import Path
import sys
import pytest
import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
from reference_sampling import elliptical_slice


def test_elliptical_slice_gaussian_conditioning():
    torch.set_num_threads(1)
    dtype = torch.float64
    mean = torch.tensor([.4, -.8], dtype=dtype)
    chol = torch.tensor([[1., 0.], [.3, .8]], dtype=dtype)
    a = torch.tensor([[1., .4], [-.2, .9]], dtype=dtype)
    y = torch.tensor([.7, -.1], dtype=dtype)
    noise = .6
    prior_precision = torch.cholesky_inverse(chol)
    covariance = torch.linalg.inv(prior_precision + a.T @ a / noise**2)
    expected = covariance @ (prior_precision @ mean + a.T @ y / noise**2)
    ll = lambda x: -.5 * ((x @ a.T - y) / noise).square().sum(-1)
    run = elliptical_slice(ll, mean, chol, seeds=[2, 3, 5, 7], draws=3000, warmup=500)
    samples = run['samples'].reshape(-1, 2)
    torch.testing.assert_close(samples.mean(0), expected, atol=.035, rtol=0)
    torch.testing.assert_close(torch.cov(samples.T), covariance, atol=.025, rtol=0)
    assert run['samples'].shape == (4, 3000, 2)
    assert run['likelihood_chain_evaluations'] >= 4 * 3501


def test_reference_reproducible_prior_and_input_ownership():
    mean = torch.zeros(2, dtype=torch.float64)
    chol = torch.eye(2, dtype=torch.float64)
    initial = torch.zeros(2, 2, dtype=torch.float64)
    kwargs = dict(seeds=[11, 13], draws=30, warmup=5, initial=initial)
    ll = lambda x: torch.zeros(x.shape[0], dtype=x.dtype)
    one = elliptical_slice(ll, mean, chol, **kwargs)['samples']
    two = elliptical_slice(ll, mean, chol, **kwargs)['samples']
    assert torch.equal(one, two)
    assert not torch.equal(one[0], one[1])
    assert torch.count_nonzero(initial) == 0
    assert torch.count_nonzero(one[:, 1:] - one[:, :-1]) > 0
    with pytest.raises(ValueError, match='distinct'):
        elliptical_slice(ll, mean, chol, seeds=[1, 1], draws=1)


def test_prior_predictive_full_grid_generation():
    from nonlinear_calibration import prior_predictive_problem
    for n in (3, 4):
        p, replicate = prior_predictive_problem(n, seed=43)
        q, other = prior_predictive_problem(n, seed=43)
        changed, _ = prior_predictive_problem(n, seed=44)
        assert p.dimension == n*n
        assert p.truth.shape == (n*n,)
        assert torch.equal(p.truth, q.truth)
        assert torch.equal(p.observations, q.observations)
        assert torch.equal(replicate, other)
        assert not torch.equal(p.truth, changed.truth)
        assert not torch.equal(p.observations, replicate)
        generator = torch.Generator().manual_seed(43)
        expected = p.offset + p.prior_scale * torch.randn(n*n, generator=generator, dtype=p.dtype)
        assert torch.equal(p.truth, expected)
        x = p.truth.repeat(2, 1)
        expected_potential = .5*((p.forward(x)-p.observations)/p.noise_scale).square().sum(-1) + .5*((x-p.offset)/p.prior_scale).square().sum(-1)
        torch.testing.assert_close(p.potential(x), expected_potential)


def test_wilson_and_dataset_coverage_units():
    from nonlinear_calibration import wilson_interval, aggregate_coverage
    low, high = wilson_interval(8, 8)
    assert .67 < low < .68 and high == pytest.approx(1.)
    for n in (3, 10):
        assert wilson_interval(n, n)[1] == 1.
        assert wilson_interval(0, n)[0] == 0.
    events = torch.tensor([[True, True], [False, True]])
    report = aggregate_coverage(events)
    assert report['independent_datasets'] == 2
    assert report['coordinate_coverage'] == [.5, 1.]
    assert report['field_fraction_mean'] == .75
    assert report['simultaneous_field_coverage'] == .5
