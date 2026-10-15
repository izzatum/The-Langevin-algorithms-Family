"""Independent-chain diagnostics checked against frozen ArviZ/SciPy results."""
import json
from pathlib import Path

import pytest
import torch

import lafa.diagnostics as diagnostics


FUNCTIONS = ['bulk_effective_sample_size', 'tail_effective_sample_size', 'mcse_mean', 'mcse_quantile']


def test_chain_quality_public_api():
    for name in FUNCTIONS:
        assert callable(getattr(diagnostics, name, None)), name


@pytest.fixture(scope='module')
def reference():
    return json.loads((Path(__file__).parent / 'data/chain_quality_arviz_0_22.json').read_text())['cases']


@pytest.mark.parametrize('name', ['iid', 'ar1', 'ties', 'heavy_tail', 'antithetic', 'shifted'])
def test_arviz_frozen_reference(reference, name):
    case = reference[name]
    x = torch.tensor(case['chains'], dtype=torch.float64)
    bulk = diagnostics.bulk_effective_sample_size(x)
    tail = diagnostics.tail_effective_sample_size(x)
    torch.testing.assert_close(bulk, x.new_tensor(case['bulk']), rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(tail, x.new_tensor(case['tail']), rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(diagnostics.mcse_mean(x), x.new_tensor(case['mean_mcse']), rtol=2e-10, atol=2e-10)
    for j, p in enumerate([.01, .05, .5, .95, .99]):
        expected = x.new_tensor([float('nan') if row[j] is None else row[j] for row in case['quantile_mcse']])
        torch.testing.assert_close(diagnostics.mcse_quantile(x, p), expected, rtol=2e-10, atol=2e-10, equal_nan=True)
    assert (bulk <= 2 * x.shape[0] * (x.shape[1] // 2)).all()


@pytest.mark.parametrize('name', FUNCTIONS)
def test_constant_dimension_is_undefined_without_poisoning_other_dimensions(name):
    x = torch.randn(3, 80, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(73))
    x[:, :, 0] = 42
    result = getattr(diagnostics, name)(x)
    assert torch.isnan(result[0])
    assert torch.isfinite(result[1]) and result[1] > 0


@pytest.mark.parametrize('name', FUNCTIONS)
def test_short_chains_and_invalid_input(name):
    fn = getattr(diagnostics, name)
    for x in [*[torch.ones(2, n, 1) for n in range(3, 10)], torch.ones(2, 8), torch.ones(2, 8, 1, dtype=torch.int64),
              torch.full((2, 16, 1), float('nan')), torch.full((2, 16, 1), float('inf'))]:
        with pytest.raises(ValueError):
            fn(x)
    x = torch.arange(10., dtype=torch.float64).reshape(1, 10, 1)
    assert fn(x).shape == (1,)  # minimum accepted input, not a convergence guarantee


@pytest.mark.parametrize('probability', [0, 1, -.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_quantile_probabilities(probability):
    x = torch.randn(2, 16, 1)
    with pytest.raises(ValueError):
        diagnostics.mcse_quantile(x, probability)


@pytest.mark.parametrize('probability', [0, .5, 1, -.1, float('nan'), True])
def test_invalid_tail_probabilities(probability):
    with pytest.raises(ValueError):
        diagnostics.tail_effective_sample_size(torch.randn(2, 16, 1), probability=probability)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_dtype_device_and_no_gradient(reference, dtype, device):
    x = torch.tensor(reference['ar1']['chains'], dtype=dtype, device=device, requires_grad=True)
    for name in FUNCTIONS:
        result = getattr(diagnostics, name)(x)
        assert result.dtype == dtype and result.device == x.device and not result.requires_grad
        baseline = getattr(diagnostics, name)(x.detach().cpu().double())
        torch.testing.assert_close(result.cpu().double(), baseline, rtol=3e-5, atol=1e-5)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_extreme_scale_equivariance(reference, scale):
    x = torch.tensor(reference['ar1']['chains'], dtype=torch.float64)
    for name in FUNCTIONS:
        baseline = getattr(diagnostics, name)(x)
        result = getattr(diagnostics, name)(x * scale)
        if name.startswith('mcse'):
            result = result / scale
        torch.testing.assert_close(result, baseline, rtol=2e-10, atol=2e-10)


def test_tied_quantile_indicator_is_undefined():
    x = torch.zeros(2, 20, 1, dtype=torch.float64)
    x[:, 0] = 1
    # The .99 quantile is the maximum, so every draw is <= its threshold.
    assert torch.isnan(diagnostics.mcse_quantile(x, .99)).all()
    assert torch.isnan(diagnostics.tail_effective_sample_size(x, probability=.01)).all()


def test_minimum_valid_bulk_ess_uses_autocorrelation():
    x = torch.arange(10., dtype=torch.float64).reshape(1, 10, 1)
    # At shorter split lengths ArviZ's finite-lag convention always gives N.
    # The accepted minimum includes positive dependence rather than that cap.
    assert diagnostics.bulk_effective_sample_size(x)[0] < 5


@pytest.mark.parametrize('dtype,scale', [(torch.float32, 3e38), (torch.float64, 1.7e308)])
def test_rhat_folding_preserves_extreme_finite_scale(dtype, scale):
    rng = torch.Generator().manual_seed(19)
    for _ in range(4):
        x = torch.randn(4, 30, 1, generator=rng, dtype=torch.float64).abs() * .1 + .65
        mask = torch.rand(4, 30, 1, generator=rng) < .2
        negative = torch.rand(4, 30, 1, generator=rng, dtype=torch.float64) * .5 - 1
        x = torch.where(mask, negative, x).clamp(-1, 1)
    x = x.to(dtype)
    expected = diagnostics.rank_normalized_split_rhat(x)
    actual = diagnostics.rank_normalized_split_rhat(x * scale)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
