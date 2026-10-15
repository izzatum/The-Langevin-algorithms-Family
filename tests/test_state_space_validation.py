"""Independent Kalman references reject invalid inputs before numerical work."""
import pytest
import torch

from lafa.problems import LinearGaussianStateSpace


@pytest.mark.parametrize('parameter', ['transition', 'observation', 'initial_mean'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_state_space_rejects_nonfinite_model_parameters(parameter, bad):
    value = [bad] if parameter == 'initial_mean' else [[bad]]
    with pytest.raises(ValueError, match='finite'):
        LinearGaussianStateSpace(**{parameter: value})


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_kalman_rejects_nonfinite_observations(bad):
    model = LinearGaussianStateSpace()
    with pytest.raises(ValueError, match='finite'):
        model.kalman_filter([[bad]])


@pytest.mark.parametrize('kwargs', [
    {'transition': torch.empty(0, 0)},
    {'observation': torch.empty(0, 1)},
])
def test_state_space_requires_positive_state_and_observation_dimensions(kwargs):
    with pytest.raises(ValueError):
        LinearGaussianStateSpace(**kwargs)


@pytest.mark.parametrize('dtype', [torch.int64, torch.bool, torch.complex64, torch.float16])
def test_state_space_requires_supported_real_floating_dtype(dtype):
    with pytest.raises(ValueError, match='dtype'):
        LinearGaussianStateSpace(dtype=dtype)


@pytest.mark.parametrize('parameter', ['transition', 'observation', 'initial_mean', 'initial_covariance', 'process_covariance', 'observation_covariance'])
@pytest.mark.parametrize('kind', ['sparse', 'complex'])
def test_state_space_rejects_unsupported_parameter_layout_or_values(parameter, kind):
    value = torch.ones(1) if parameter == 'initial_mean' else torch.ones(1, 1)
    value = value.to_sparse() if kind == 'sparse' else value.to(torch.complex128) * (1+1j)
    with pytest.raises(ValueError, match='dense|real'):
        LinearGaussianStateSpace(**{parameter: value})


@pytest.mark.parametrize('value', [torch.ones(1, 1).to_sparse(), torch.tensor([[1+1j]])])
def test_kalman_rejects_sparse_or_complex_observations(value):
    with pytest.raises(ValueError, match='dense|real'):
        LinearGaussianStateSpace().kalman_filter(value)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_state_space_retains_valid_dtype_input_ownership_and_reference_identity(dtype):
    mean = torch.tensor([.2], dtype=dtype)
    model = LinearGaussianStateSpace(initial_mean=mean, dtype=dtype)
    mean.zero_()
    y = torch.tensor([[.8]], dtype=dtype)
    result = model.kalman_filter(y)
    expected = torch.tensor([[.2+(.8-.2)/1.2]], dtype=dtype)
    torch.testing.assert_close(result.means, expected)
    torch.testing.assert_close(y, torch.tensor([[.8]], dtype=dtype))
    assert result.means.dtype == dtype and torch.isfinite(result.log_likelihood)
    states1, observations1 = model.simulate(3, seed=42)
    states2, observations2 = model.simulate(3, seed=42)
    torch.testing.assert_close(states1, states2, rtol=0, atol=0)
    torch.testing.assert_close(observations1, observations2, rtol=0, atol=0)
